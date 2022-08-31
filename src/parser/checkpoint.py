import os
from .base import ArgumentParser, adding_cuda
from .tools import load_args


def parser():
    parser = ArgumentParser()
    parser.add_argument("checkpointname")
    
    opt = parser.parse_args()
    
    folder, checkpoint = os.path.split(opt.checkpointname)
    parameters = load_args(os.path.join(folder, "opt.yaml"))

    adding_cuda(parameters)
    epoch = int(checkpoint.split("_")[-1].split('.')[0])
    return parameters, folder, checkpoint, epoch


def construct_checkpointname(parameters, folder):
    implist = [parameters["modelname"],
               parameters["dataset"],
               parameters["extraction_method"],
               parameters["pose_rep"]]
    if parameters["pose_rep"] != "xyz":
        # [True, ""] to be compatible with generate job
        if "glob" in parameters:
            implist.append("glob" if parameters["glob"] in [True, ""] else "noglob")
        else:
            implist.append("noglob")
        if "translation" in parameters:
            implist.append("translation" if parameters["translation"] in [True, ""] else "notranslation")
        else:
            implist.append("notranslation")
            
        if "rcxyz" in parameters["modelname"]:
            implist.append("joinstype_{}".format(parameters["jointstype"]))

    if "num_layers" in parameters:
        implist.append("numlayers_{}".format(parameters["num_layers"]))
            
    for name in ["num_frames", "min_len", "max_len", "num_seq_max"]:
        pvalue = parameters[name]
        pname = name.replace("_", "")
        if pvalue != -1:
            implist.append(f"{pname}_{pvalue}")
    
    if "view" in parameters:
        if parameters["view"] == "frontview":
            implist.append("frontview")

    if "use_z" in parameters:
        if parameters["use_z"] != 0:
            implist.append("usez")
        else:
            implist.append("noz")

    if "vertstrans" in parameters:
        implist.append("vetr" if parameters["vertstrans"] else "novetr")
        
    if "ablation" in parameters:
        abl = parameters["ablation"]
        if abl not in ["", None]:
            implist.append(f"abl_{abl}")
            
    if parameters["num_frames"] != -1:
        implist.append("sampling_{}".format(parameters["sampling"]))
        if parameters["sampling"] == "conseq":
            implist.append("samplingstep_{}".format(parameters["sampling_step"]))
    if "lambda_kl" in parameters:
        implist.append("kl_{:.0e}".format(float(parameters["lambda_kl"])))

    if "activation" in parameters:
        act = parameters["activation"]
        implist.append(act)

    implist.append("bs_{}".format(parameters["batch_size"]))
    implist.append("ldim_{}".format(parameters["latent_dim"]))
    
    checkpoint = "_".join(implist)
    return os.path.join(folder, checkpoint)


