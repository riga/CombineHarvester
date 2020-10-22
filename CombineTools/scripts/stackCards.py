#!/usr/bin/env python
# coding: utf-8

"""
Datacard stacking tool.
Unlike combineCards.py which merges the bins of multiple datacards "horizontally", i.e., it joins
the collections of bins across all input datacards, this tool stacks input yields and histograms
"vertically", effectively adding yields and bin contents. This implies that, per process, bin edges
in input histograms must be identical. Nuisances with the same name are considered to be fully
correlated and their combined effect is calculated following bin-wise Gaussian error propagation.
Supported systematic types: lnN, lnU, rateParam, shape*.
"""


__author__ = "Marcel Rieger"
__email__ = "marcel.rieger@cern.ch"
__version__ = "0.1.0"


import os
import sys
import re
import tempfile
from glob import glob
from fnmatch import fnmatch
from collections import OrderedDict


# import custom modules lazily
ROOT = None
ch = None

# constants
DATA_OBS = "data_obs"


class DotDict(dict):
    """
    Class inheriting from the standard dict class that enables read-access on items via attributes.
    """

    def __getattr__(self, attr):
        return self[attr]


def lazy_imports():
    """
    Imports ROOT and the CombineHarvester modules lazily into the global scope. This allows starting
    this script (e.g. with *--help*) in any environment.
    """
    global ROOT
    global ch

    if ROOT is None:
        import ROOT
        ROOT.PyConfig.IgnoreCommandLineOptions = True
        ROOT.gROOT.SetBatch()

    if ch is None:
        import CombineHarvester.CombineTools.ch as ch


def stack_cards(input_cards, output_card, output_shapes, analysis_name="analysis", era="13TeV",
        mass="125", channel_name="channel", bin_name=None, auto_mc_stats=None, shape_patterns=None,
        rate_digits=5, syst_digits=3, output_directory="./"):
    """
    Actual function that performs the datacard stacking. For more information on arguments, run this
    script with *--help* or check out the setup of the argument parser below.
    """
    lazy_imports()

    #
    # prepare inputs
    #

    input_data = []

    # extract optional bin names in input card strings and expand globbing statements
    inputs = []
    for card in input_cards:
        card_bin_name, path = re.match(r"^(([^\/]+)=)?(.+)$", card).groups()[1:]
        inputs.extend([(_path, card_bin_name) for _path in glob(path)])

    # stop here when no valid inputs exist
    if not inputs:
        print("no matching datacards found")
        sys.exit(1)

    # loop through cards and setup the input data structure
    for path, card_bin_name in inputs:
        path = os.path.expandvars(os.path.expanduser(str(path)))

        # store data
        input_data.append(DotDict(
            path=path,
            abs_path=os.path.normpath(os.path.abspath(path)),
            cb=None,
            bin_name=card_bin_name,
            observation=None,
            processes=OrderedDict(),  # processes by name
            systematics=[],  # systematics objects
            systematic_names=[],  # strings
            parameters=OrderedDict(),  # parameters by name
            auto_mc_stats=None,  # 3-tuple (threshold, signal, hist)
            groups=OrderedDict(),  # parameter names by group name
            rate_param_infos=[],  # 4-tuples (name, bin, process, rest)
        ))

    #
    # parse inputs
    #

    # initialize CombineHarvester objects and parse cards
    for data in input_data:
        data["cb"] = cb = ch.CombineHarvester()
        cb.SetVerbosity(0)

        # parse
        cb.ParseDatacard(data.path, analysis_name, era, channel_name, 1, mass)

        # check if the requested bin name exists or, if none was set, complain when there is more
        # than one bin
        bin_names = cb.bin_set()
        if data.bin_name:
            if data.bin_name not in bin_names:
                raise Exception("requested bin name '{}' for datacard at {} does not exist".format(
                    data.bin_name, data.path))
            # select the bin
            data["cb"] = cb = filter_cb(cb, bin_name=data.bin_name)
        elif len(bin_names) != 1:
            raise Exception("this tool requires input datacards to have exactly one bin when no "
                "name is specified for that card, but found {} bins in datacard {}: {}".format(
                    len(bin_names), data.path, bin_names))
        else:
            data["bin_name"] = bin_names[0]

        # store the observation
        observations = []
        cb.ForEachObs(observations.append)
        data["observation"] = observations[0]

        # store processes
        def add_process(process):
            data.processes[process.process()] = process
        cb.ForEachProc(add_process)

        # store systematics and their names
        cb.ForEachSyst(data.systematics.append)
        data["systematic_names"] = cb.syst_name_set()

        # store parameters referred to by systematics
        for syst in data.systematics:
            if syst.name() not in data.parameters:
                parameter = cb.GetParameter(syst.name())
                if not parameter:
                    print("WARNING: no parameter found for systematic '{}'".format(syst.name()))
                else:
                    data.parameters[syst.name()] = parameter

        # save auto mc stats info
        if data.bin_name in cb.GetAutoMCStatsBins():
            data["auto_mc_stats"] = (
                cb.GetAutoMCStatsEventThreshold(data.bin_name),
                cb.GetAutoMCStatsIncludeSignal(data.bin_name),
                cb.GetAutoMCStatsHistMode(data.bin_name),
            )

        # store group infos
        for parameter in data.parameters.values():
            for group in parameter.groups():
                data.groups.setdefault(group, []).append(parameter.name())

        # extract rateParams manually and store only those matching the selected bin
        data["rate_param_infos"] = extract_rate_params(data.abs_path, bin_name=data.bin_name)

    #
    # combine input information
    #

    # combined infos across input data, try to preserve orders
    all_bin_names = []
    all_observations = []
    all_process_names = []
    all_signal_names = []
    all_background_names = []
    all_systematic_names = []
    all_auto_mc_stats = []
    all_groups = OrderedDict()
    for data in input_data:
        extend_unique(all_bin_names, [data.bin_name])
        all_observations.append(data.observation)
        extend_unique(all_process_names, list(data.processes.keys()))
        extend_unique(all_signal_names,
            [p.process() for p in data.processes.values() if p.signal()])
        extend_unique(all_background_names,
            [p.process() for p in data.processes.values() if not p.signal()])
        extend_unique(all_systematic_names, data.systematic_names)
        all_auto_mc_stats.append(data.auto_mc_stats)
        for group_name, parameter_names in data.groups.items():
            all_groups.setdefault(group_name, []).extend(parameter_names)

    # fill a systematics map {syst_name: {process_name: {input card index: syst}}} for later use
    syst_map = OrderedDict(
        (syst_name, OrderedDict(
            (process_name, OrderedDict(
                (data_key, None)
                for data_key in range(len(input_data))
            ))
            for process_name in all_process_names
        ))
        for syst_name in all_systematic_names
    )
    for data_key, data in enumerate(input_data):
        for syst in data.systematics:
            syst_map[syst.name()][syst.process()][data_key] = syst

    #
    # sanity checks and defaults
    #

    # check that, per process, systematics have the same, valid type across datacards
    rate_syst_types = ["lnN", "lnU", "rateParam"]
    for syst_name, syst_data in syst_map.items():
        for process_name, process_data in syst_data.items():
            syst_types = [(syst and syst.type()) for syst in process_data.values()]
            unique_types = list(set(syst_types) - {None})
            if not unique_types:
                continue
            elif len(unique_types) > 1:
                pairs = [
                    (input_data[data_key].path, syst_type)
                    for data_key, syst_type in zip(process_data, syst_types)
                    if syst_type
                ]
                raise Exception("types of systematic '{}' is found to vary:\n{}".format(
                    syst_name, "\n".join("  {}: {}".format(*pair) for pair in pairs)))
            elif unique_types[0] not in rate_syst_types and not unique_types[0].startswith("shape"):
                raise Exception("type '{}' of systematic '{}' is not supported, supported ones "
                    "are: {},shape*".format(unique_types[0], syst_name, ",".join(rate_syst_types)))

    # check that either all mc stats are set or none, and optionally set defaults
    if all_auto_mc_stats.count(None) == 0:
        # all datacards have autoMCStats settings
        if not auto_mc_stats:
            auto_mc_stats = tuple(str(s) for s in all_auto_mc_stats[0])
    elif all_auto_mc_stats.count(None) != len(input_data):
        # a few datacards have autoMCStats settings, but others do not
        pairs = [(data.path, m) for data, m in zip(input_data, all_auto_mc_stats)]
        raise Exception("either no input datacard should contain autoMCStats or all, but found:\n"
            + "\n".join("  {}: {}".format(*pair) for pair in pairs))

    # set a default bin name, consisting of all present, unique bin names concatenated by "__"
    if bin_name is None:
        unique_bin_names = sorted(list(set(all_bin_names)), key=lambda b: all_bin_names.index(b))
        bin_name = "__".join(unique_bin_names)

    #
    # stack shape files
    #

    tmp_shape_file = stack_shapes(input_data, syst_map, shape_patterns, bin_name, mass=mass)

    #
    # create the output datacard
    #

    # create the new CombineHarvester object
    out_cb = ch.CombineHarvester()
    out_cb.SetVerbosity(0)

    # add the observation
    out_cb.AddObservations([mass], [analysis_name], [era], [channel_name], [(1, bin_name)])

    # set its rate (possibly overwritten during shape extraction later on)
    sum_obs = sum(obs.rate() for obs in all_observations)
    set_obs = lambda out_obs: out_obs.set_rate(round(sum_obs, rate_digits))
    out_cb.ForEachObs(set_obs)

    # add signal and background processes
    out_cb.AddProcesses([mass], [analysis_name], [era], [channel_name], all_signal_names,
        [(1, bin_name)], True)
    out_cb.AddProcesses([mass], [analysis_name], [era], [channel_name], all_background_names,
        [(1, bin_name)], False)

    # set their rates (possibly overwritten during shape extraction later on)
    all_out_processes = []
    out_cb.ForEachProc(all_out_processes.append)
    for out_process in all_out_processes:
        rate = sum(get_rate(data.cb, out_process.process()) for data in input_data)
        out_process.set_rate(round(rate, rate_digits))

    # add systematics
    for syst_name, syst_data in syst_map.items():
        for process_name, process_data in syst_data.items():
            # stop if the systematic does not apply to the process
            if set(process_data.values()) == {None}:
                continue

            # get the first systematic object to access its type
            set_data_keys = [data_key for data_key, syst in process_data.items() if syst]
            first_data_key = set_data_keys[0]
            first_syst = process_data[first_data_key]
            syst_type = first_syst.type()

            # get the systematic scale (a 2-tuple with down/up factors)
            if syst_type in ["lnN", "lnU"]:
                scale_d, scale_u, sym = get_scale_lognormal(input_data, process_data, process_name,
                    syst_type)
            elif syst_type.startswith("shape"):
                # the exact scales are set in the shape histograms
                scale_d, scale_u, sym = 1., 1., True
            elif syst_type == "rateParam":
                # rateParams are special in the sense that they cannot be properly accessed or added
                # via the python bindings, so just use the configuration of the first defined one
                for rate_param_info in input_data[first_data_key].rate_param_infos:
                    if rate_param_info[0] == syst_name:
                        break
                else:
                    # this should never occur as at least one datacard must have specified the param
                    raise Exception("misconfiguration of rateParam '{}'".format(syst_name))

                out_cb.AddDatacardLineAtEnd("{} rateParam {} {} {}".format(
                    syst_name, bin_name, process_name, rate_param_info[3]))
                continue
            else:
                # following the check above, this should never happen
                raise NotImplementedError

            # apply rounding
            scale = (round(scale_d, syst_digits), round(scale_u, syst_digits))

            # add it
            # also, when the scale is symmetric, it is a combine habit to pass only the up value
            filter_cb(out_cb, process_name=process_name).AddSyst(out_cb, syst_name, syst_type,
                ch.SystMap()(scale[1] if sym else scale))

    # extract shapes
    if tmp_shape_file:
        # sort shape patterns such that entries whose process name is specific are at the front
        sorted_shape_patterns = sort_shape_patterns(shape_patterns)

        # keep a list of processes that were not yet handled, adding data manually
        processes_to_handle = set(out_cb.process_set())
        processes_to_handle.add(DATA_OBS)

        for process_name, nom_pattern, syst_pattern in sorted_shape_patterns:
            # get names of matching processes
            names = {name for name in processes_to_handle if fnmatch(name, process_name)}

            # complain when no processes are left
            if not names:
                raise Exception("the process name or pattern '{}' does not match any processes "
                    "left for extracting shapes, matching processes were either already handled or "
                    "did not exist initially".format(process_name))

            # remove from left processes
            processes_to_handle -= names

            # treat data differently
            skip_data = DATA_OBS not in names
            names -= {DATA_OBS}

            # filter processes and extract their shapes
            _out_cb = filter_cb(out_cb, process_name=list(names))
            _out_cb.ExtractShapes(tmp_shape_file, nom_pattern, syst_pattern, skip_data)

        # remove the tmp shape file
        os.remove(tmp_shape_file)

    # add autoMCStats
    if auto_mc_stats:
        threshold = float(auto_mc_stats[0])
        include_signal = auto_mc_stats[1].lower() in ("yes", "true", "1")
        hist_mode = int(auto_mc_stats[2])
        out_cb.SetAutoMCStatsByBin(bin_name, threshold, include_signal, hist_mode)

    # add groups
    for group_name, parameter_names in all_groups.items():
        out_cb.SetGroup(group_name, parameter_names)

    # write it
    output_directory = os.path.expandvars(os.path.expanduser(output_directory))
    writer = ch.CardWriter(os.path.join("$TAG", output_card), os.path.join("$TAG", output_shapes))
    writer.WriteCards(output_directory, out_cb)


def stack_shapes(input_data, syst_map, shape_patterns, bin_name, mass="125"):
    """
    Takes the *input_data* and *syst_map* objects generated by :py:func:`stack_cards`, which
    contain all information on the processes and systematics of the input datacards, reads all
    combinations of histograms and stacks them per process and systematic.

    The stacked histograms are written into a new ROOT file in a structure that is customized by a
    list of *shape_patterns*. Each pattern should be a 2- or 3-tuple with a process name (supports
    wildcards), the nominal pattern used to store nominal histograms, and an optional pattern to
    store systematically varied histograms. In these patters, the variables ``"%BIN"``,
    ``"%PROCESS"``, ``"%SYSTEMATIC"`` and ``"%MASS"`` are replaced with the values applying for that
    particular histogram. Therefore, the *bin_name* and a *mass* value need to be supplied.

    The path of the created ROOT file is returned.
    """
    lazy_imports()

    # helper to exract the visible edges of a histogram
    def get_bin_edges(hist):
        return [
            hist.GetBinLowEdge(b) for b in range(1, hist.GetNbinsX() + 1)
        ] + [hist.GetXaxis().GetXmax()]

    # helper to check if a number of histograms have the exact same bin edges
    def check_bin_edges(hists, epsilon=1e-5):
        edges = None
        for i, hist in enumerate(hists):
            _edges = get_bin_edges(hist)
            if not edges:
                edges = _edges
            elif len(_edges) != len(edges):
                # overall number of bins does not match
                raise Exception("histogram {} has {} bins, but should be {}".format(
                    hist.GetName(), len(_edges) - 1, len(edges) - 1))
            else:
                # compare each bin edge separately
                for b, (e1, e2) in enumerate(zip(edges, _edges)):
                    if abs(e1 - e2) <= epsilon:
                        continue
                    raise Exception("bin edge {} of histogram {} is at {}, but should be {}".format(
                        b, hist.GetName(), e2, e1))
        return edges

    # helper to replace variables in strings
    def replace_vars(s, process_name, syst_name=None):
        s = s.replace("$PROCESS", process_name).replace("$BIN", bin_name).replace("$MASS", mass)
        if syst_name:
            s = s.replace("$SYSTEMATIC", syst_name)
        return s

    # get a set of all process names to be handled, manually include data_obs
    processes_to_handle = set(sum((list(data.processes.keys()) for data in input_data), []))
    processes_to_handle.add(DATA_OBS)

    # book a temporary file and open it
    tmp_file = tempfile.mkstemp(suffix=".root")[1]
    tfile = ROOT.TFile(tmp_file, "RECREATE")

    # sort the shape patterns so that those with specific process names are in front, then loop
    for shape_pattern in sort_shape_patterns(shape_patterns):
        process_pattern, nom_pattern, syst_pattern = (shape_pattern + [None])[:3]

        # get names of matching processes
        process_names = [name for name in processes_to_handle if fnmatch(name, process_pattern)]

        # do nothing when no processes are left
        if not process_names:
            continue

        # remove from processes left to handle
        processes_to_handle -= set(process_names)

        for process_name in process_names:
            # stack nominal histograms first
            nom_hists = []
            for data in input_data:
                if process_name == DATA_OBS:
                    # observation
                    nom_hists.append(data.cb.GetObservedShape())
                elif process_name in data.processes:
                    # normal processes
                    nom_hists.append(get_shape(data.cb, process_name, with_uncertainty=True))
                else:
                    # missing process for that input datacard
                    nom_hists.append(None)
            real_nom_hists = [hist for hist in nom_hists if hist]

            # check that the binning is identical
            check_bin_edges(real_nom_hists)

            # ensure that intermediate directories exist
            dst = replace_vars(nom_pattern, process_name)
            dir_name, hist_name = os.path.split(dst)
            tdir = tfile.Get(dir_name)
            if not tdir:
                tdir = tfile.mkdir(dir_name)
            tdir.cd()

            # ensure that the observation is always called data_obs
            if process_name == DATA_OBS:
                hist_name = process_name

            # stack them
            nom_hist = real_nom_hists[0].Clone(hist_name)
            for hist in real_nom_hists[1:]:
                nom_hist.Add(hist)

            # write it
            nom_hist.Write()

            # stack systematic hists now, so nothing to do for data or when no syst_pattern exists
            if process_name == DATA_OBS or not syst_pattern:
                continue

            for syst_name, syst_data in syst_map.items():
                # first, check if the process has systematics at all
                if process_name not in syst_data:
                    continue
                process_data = syst_data[process_name]
                if all(syst is None for syst in process_data.values()):
                    continue

                # do nothing for non-shape systematics
                first_syst = filter(bool, list(process_data.values()))[0]
                if not first_syst.type().startswith("shape"):
                    continue

                # get systematicly varied hists, and in case a systematic does not apply for a
                # process in a certain datacard, take the nominal one
                for direction in ["up", "down"]:
                    syst_hists = []
                    for i, (data, syst) in enumerate(zip(input_data, process_data.values())):
                        if syst:
                            syst_hists.append(get_shape(data.cb, process_name,
                                with_uncertainty=True, variation=(syst, direction)))
                        else:
                            syst_hists.append(nom_hists[i])

                    # check that the binning is identical
                    check_bin_edges(syst_hists)

                    # ensure that intermediate directories exist
                    dst = replace_vars(syst_pattern, process_name, syst_name=syst_name)
                    dir_name, hist_name = os.path.split(dst)
                    tdir = tfile.Get(dir_name)
                    if not tdir:
                        tdir = tfile.mkdir(dir_name)
                    tdir.cd()

                    # stack them
                    syst_hist = syst_hists[0].Clone(hist_name + direction.capitalize())
                    for hist in syst_hists[1:]:
                        syst_hist.Add(hist)

                    # write it
                    syst_hist.Write()

    # finalize
    tfile.Close()

    return tmp_file


def get_scale_lognormal(input_data, process_data, process_name, syst_type):
    """
    Computes the scale of a systematic with type *lnN* or *lnU* given in *process_data*, acting on a
    process defined by *process_name*, combined for multiple datacards given in *input_data*. See
    :py:func:`stack_cards` for more info on the structure of *input_data*. *process_data* should be
    a dictionary mapping keys of the respective entry in *input_data* to ch.Systematic instances and
    *syst_type* is the type of the systematic in question.

    For both *lnN* and *lnU*, the combined scales can be understood as the mean of the scales of the
    initial datacards, weighted by the respective yield of the relevant process. The difference in
    the meaning of *lnN* and *lnU* systematics is accounted for (i.e. relative, 1-sigma
    uncertainties ``dx/x`` are encoded as ``1-dx/x / 1+dx/x`` for *lnN* whereas for *lnU* the
    relative, uniform range around the central value ``[e/1-e, e]`` is written as ``1-e / 1+e``).

    The return value is a 3-tuple containing the down scale, up scale, and a flag that denotes
    whether the scale itself is symmetric.
    """
    # this should never match
    assert(syst_type in ("lnN", "lnU"))

    # store the sum of weighted mean numerators and the denominator (i.e. the sum of rates)
    # common to both directions and a sym flag that is True when all syst objects were symmetric
    num_d = 0.
    num_u = 0.
    den = 0.
    sym = True

    # go through datacards and add values
    for data_key, syst in process_data.items():
        # get the CombineHarvester object
        cb = input_data[data_key].cb

        # add denominator
        rate = get_rate(cb, process_name)
        den += rate

        # no contribution to the numerators when the syst is not defined
        if syst:
            # update the sym flag
            _sym = not syst.asymm()
            sym &= _sym

            # get relative down and up variations
            # when symmetric, the habit in combine is to use the inverted up value
            var_d = (1. - syst.value_d()) if not _sym else (syst.value_u() - 1.)
            var_u = syst.value_u() - 1.

            # account for different meaning of lnU down variation which is e/1-e and
            # only approximately e for small e, so account for this
            if syst_type == "lnU":
                var_d /= 1. - var_d

            # amend numerators
            num_d += var_d * rate
            num_u += var_u * rate

    # divide to obtain weighted means
    rel_d = (num_d / den) if den else 0.
    rel_u = (num_u / den) if den else 0.

    # convert to type-dependent return types
    if syst_type == "lnN":
        scale_d = 1. - rel_d
        scale_u = 1. + rel_u
    else:  # lnU
        scale_d = 1. / (1. + rel_d)
        scale_u = 1. + rel_u

    return (scale_d, scale_u, sym)


def get_systematic(cb, syst_name, process_name=None, silent=False):
    """
    Returns the ch.Systematic instance named *syst_name* from a ch.CombineHarvester object *ch*.
    When *process_name* is set, the systematic's process must match that name. When no systematic
    was found, an exception is raised unless *silent* is *True*. In that case, *None* is returned.
    """
    # extract all systematics
    systs = []
    cb.ForEachSyst(systs.append)

    # find the correct one
    for syst in systs:
        if syst.name() == syst_name and (process_name is None or syst.process() == process_name):
            return syst

    # silently return None or raise
    if silent:
        return None

    msg = "could not find systematic '{}'".format(syst_name)
    if process_name:
        msg += " applying to process '{}'".format(process_name)
    raise Exception(msg)


def get_rate(cb, process_name):
    """
    Returns the nominal rate of a process defined by *process_name* from a ch.CombineHarvester
    instance *cb*.
    """
    # select the process
    cb = filter_cb(cb, process_name=process_name)

    # return the rate
    return cb.GetRate()


def get_shape(cb, process_name, with_uncertainty=False, variation=None, silent=False):
    """
    Extracts a shape as a ROOT.TH1F object from a ch.CombineHarvester instance *cb* for process
    defined by *process_name*. When *with_uncertainty* is *True*, the returned shape contains bin
    errors from the combination of all applying systematics. When *variation* is set, it should be a
    2-tuple with either a ch.Systematic instance or the name of one as the first element, and the
    direction (``"up"`` or ``"down"``) as the second element. In case a name is given and the
    corresponding systematic could not be found, an error is raised or *None* is returned when
    *silent* is *True*. Otherwise, the varied rather than the nominal shape is returned.
    """
    # select the process
    cb = filter_cb(cb, process_name=process_name)

    # get the nominal shape
    nominal_shape = cb.GetShapeWithUncertainty() if with_uncertainty else cb.GetShape()

    # when a variation is defined, get the normalized (!) shape variation and rescale it
    if variation:
        syst, direction = variation
        if direction not in ("up", "down"):
            raise ValueError("when a variation is set, its second tuple element must be either "
                "'up' or 'down', but got '{}'".format(direction))

        # when syst is a name, look up the proper syst object
        # otherwise, the passed process name must match the process name of the systematic
        if not isinstance(syst, ch.Systematic):
            syst = get_systematic(cb, syst, process_name=process_name, silent=silent)
        elif syst.process() != process_name:
            raise ValueError("passed process name '{}' does not match '{}' described by the "
                "systematic '{}'".format(process_name, syst.process(), syst.name()))

        shape = syst.ShapeUAsTH1F() if direction == "up" else syst.ShapeDAsTH1F()
        scale = syst.value_u() if direction == "up" else syst.value_d()
        shape.Scale(nominal_shape.Integral() * scale)

        return shape
    else:
        return nominal_shape


def extract_rate_params(path, bin_name=None):
    """
    Extracts information on *rateParam* systematics contained in a datacard at *path*. The return
    value is a list of 4-tuples containing the parameter name, the bin, the process and the rest of
    the line as a string. When a *bin_name* is given, parameters are skipped that do not apply to
    the referred bin.
    """
    path = os.path.expandvars(os.path.expanduser(path))

    # read the lines
    with open(path, "r") as f:
        lines = f.readlines()

    rate_params = []

    # go through lines and identify systematics block
    # (trigger on 4 lines starting with 'bin', 'process', 'process', 'rate' in that exact order)
    trigger_words = ("bin", "process", "process", "rate")
    first_words = []
    in_systematics = False
    for line in lines:
        # skip comments and empty lines
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # split into words
        words = re.sub(r"\s+", " ", line).split(" ")
        # check if we are already in the systematics block
        if not in_systematics:
            # add the first word
            first_words.append(re.sub(r"\s+", " ", line).split(" ")[0])
            # do the last four first words trigger?
            if tuple(first_words[-4:]) == trigger_words:
                in_systematics = True
                continue
        elif len(words) >= 4 and words[1] == "rateParam":
            # here, we are dealing with a rate param
            rate_params.append((words[0], words[2], words[3], " ".join(words[4:])))

    # filter bins
    if bin_name:
        rate_params = [
            params for params in rate_params
            if fnmatch(bin_name, params[1])
        ]

    return rate_params


def filter_cb(cb, analysis_name=None, era_name=None, mass=None, channel_name=None, bin_name=None,
        process_name=None):
    """
    Applies several filters to the copy of a ch.CombineHarvester object *cp* and returns it.
    """
    cb = cb.cp()
    if analysis_name:
        cb = cb.analysis(make_list(analysis_name))
    if era_name:
        cb = cb.era(make_list(era_name))
    if mass:
        cb = cb.mass(make_list(mass))
    if channel_name:
        cb = cb.channel(make_list(channel_name))
    if bin_name:
        cb = cb.bin(make_list(bin_name))
    if process_name:
        cb = cb.process(make_list(process_name))
    return cb


def sort_shape_patterns(shape_patterns):
    """
    Sorts a list of *shape_patterns* (tuples) such that those patterns are moved towards the end,
    where the first element, i.e. the process name, contains wildcard (or pattern) expressions. The
    less non-pattern characters a process name has, the higher the index in the returned list.
    """
    def sort_fn(tpl):
        process_name = tpl[0]
        if is_pattern(process_name):
            non_pattern_chars = process_name.replace("*", "").replace("?", "")
            return (1000 - len(non_pattern_chars)) * 1000
        else:
            return shape_patterns.index(tpl)

    return sorted(shape_patterns, key=sort_fn)


def extend_unique(dst, *srcs):
    """
    Extends a list *dst* with other lists given by *srcs* while preserving uniqueness of elements.
    """
    for src in srcs:
        for elem in src:
            if elem not in dst:
                dst.append(elem)


def make_list(l):
    """
    When *l* is not a list, a new list with *l* as its only element is returned. Otherwise, a copy
    of *l* is returned.
    """
    return list(l) if isinstance(l, list) else [l]


def is_pattern(s):
    """
    Returns *True* when the string *s* contains pattern characters (``"?"`` and ``"*"``) and *False*
    otherwise. Sequence patterns are not interpreted.
    """
    return "*" in s or "?" in s


def main():
    import argparse

    # custom argument types
    csv = lambda value: value.strip().split(",")

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", "-i", nargs="+", metavar="(BIN=)DATACARD", help="paths to "
        "datacards to stack with an optional bin name to select a bin in case the datacard "
        "contains multiple; without specifying a bin, the datacard must have exactly one bin")
    parser.add_argument("--outputs", "-o", nargs=2, help="two paths to save the output datacard "
        "and shapes file", metavar=("DATACARD", "SHAPE_FILE"))
    parser.add_argument("--directory", "-d", default=os.getcwd(), help="an optional output "
        "directory; default: ./")
    parser.add_argument("--analysis", "-a", default="analysis", help="the analysis name; default: "
        "analysis")
    parser.add_argument("--era", "-e", default="13TeV", help="the analysis era; default: 13TeV")
    parser.add_argument("--mass", "-m", default="125", help="the mass attribute; default: 125")
    parser.add_argument("--channel", "-c", default="channel", help="name of the output channel; "
        "default: channel")
    parser.add_argument("--bin", "-b", default=None, help="name of the output bin; default: "
        "input bin names concatenated with two underscores")
    parser.add_argument("--auto-mc-stats", nargs=3, metavar=("THRESHOLD", "SIGNAL", "HIST_MODE"),
        default=None, help="threshold, signal flag (usually 0) and hist mode (usually 1) for the "
        "autoMCStats setting; default: setting of the first datacard")
    parser.add_argument("--shape-patterns", "-s", type=csv, nargs="+", metavar="PROCESS,NOM_PATTERN"
        "(,SYST_PATTERN)", default=[["*", "$BIN/$PROCESS", "$BIN/$PROCESS_$SYSTEMATIC"]],
        help="patterns for saving stacked shapes; the systematic pattern is optional; per process "
        "the most specific pattern is used; accepts variables %%BIN, %%PROCESS, %%MASS and "
        "%%SYSTEMATIC; default: *,$BIN/$PROCESS,$BIN/$PROCESS_$SYSTEMATIC")
    parser.add_argument("--rate-digits", type=int, default=5, help="number of digits of combined "
        "rates, default: 5")
    parser.add_argument("--syst-digits", type=int, default=3, help="number of digits of combined "
        "systematics, default: 3")
    args = parser.parse_args()

    # run the stacking
    stack_cards(args.inputs, args.outputs[0], args.outputs[1], analysis_name=args.analysis,
        era=args.era, mass=args.mass, channel_name=args.channel, bin_name=args.bin,
        auto_mc_stats=args.auto_mc_stats, shape_patterns=args.shape_patterns,
        rate_digits=args.rate_digits, syst_digits=args.syst_digits, output_directory=args.directory)


# entry hook
if __name__ == "__main__":
    main()
