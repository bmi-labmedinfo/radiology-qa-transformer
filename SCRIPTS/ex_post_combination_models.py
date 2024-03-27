# EX-POST COMBINATION MODELS + VALIDATION
import pandas as pd
import numpy as np
from configuration import Config
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from metrics import validate
from dataset_creation import *
import re
from metrics import *
config = Config()

class ExPostCombination_PretrainedModel:
    def __init__(self, k_fold=0, seed=config.SEEDS[0]):
        self.k = k_fold
        self.seed = seed
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_CHECKPOINT)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)
        config.USE_PRETRAINED = True
        config.POST_PROCESSING = False
        self.res_dir = config.RESULTS_DIR + "ex-post-comb/pretrained/"
        # DATASET PREPARATION
        self.multitask_dataset = epc_k_fold_splitting(self.k, seed, self.tokenizer)
        self.sample_test = self.multitask_dataset["test"].shuffle(seed)
        self.sample_test = self.sample_test.map(lambda example: {"output_text": example["output_text"].lower()})

    def first_validation(self):
        print("> EPC_pretrained_model (fold", self.k, "): FIRST VALIDATION START")
        unconst_res = validate(self.model, self.tokenizer, self.sample_test, constrained=False, verbose=True)
        for r in unconst_res["metric"]:
            unconst_res["metric"][r]["fold"] = self.k
        unc_df = pd.DataFrame(unconst_res["metric"])
        print(unc_df)
        unc_df.to_csv(self.res_dir + "EPC_PT_unc_results_batches_" + str(self.k) + ".csv")
        print(f"EPC_PT_unc_results_batches {self.k} fold: SAVED!")

        const_res = validate(self.model, self.tokenizer, self.sample_test, constrained=True, verbose=True, k_fold=self.k)
        for r in const_res["metric"]:
            const_res["metric"][r]["fold"] = self.k
        con_df = pd.DataFrame(const_res["metric"])
        print(con_df)
        con_df.to_csv(self.res_dir + "EPC_PT_con_results_batches_" + str(self.k) + ".csv")
        print(f"EPC_PT_unc_results_batches {self.k} fold: SAVED!")

        pretrained_table = pd.DataFrame({"id": self.sample_test["id"],
                                         "batches": self.sample_test["batches"],
                                         "type": self.sample_test["type"],
                                         "input_text": self.sample_test["input_text"],
                                         "ref": self.sample_test["output_text"],
                                         "unc_pred": unconst_res["predictions"],
                                         "con_pred": const_res["predictions"],
                                         "unc_conf": unconst_res["confidences"],
                                         "con_conf": const_res["confidences"]})
        pretrained_table.to_csv(self.res_dir + "EPC_PT_table_batches_" + str(self.k) + ".csv")
        print(f"EPC_PT_table_batches {self.k} fold: SAVED!")
        print("> EPC_pretrained_model (fold", self.k, "): FIRST VALIDATION END")

    def reassembly_batches(self):
        print("> EPC_pretrained_model (fold", self.k, "): REASSEMBLY OF BATCHES START")
        tag = "PT"
        conf_theshold = 0.2
        sample_table = pd.read_csv(self.res_dir + "EPC_PT_table_batches_" + str(self.k) + ".csv")

        unique_ids = list(sample_table['id'].unique())
        list_type = []
        list_ref = []
        list_input = []
        list_pred_unc = []
        list_pred_con = []

        for id in np.arange(len(unique_ids)):
            row_idx = list(sample_table.index[sample_table['id'] == unique_ids[id]])
            list_type.append(sample_table.iloc[row_idx[0], list(sample_table.columns).index("type")])
            list_ref.append(sample_table.iloc[row_idx[0], list(sample_table.columns).index("ref")])
            list_input.append(sample_table.iloc[row_idx[0], list(sample_table.columns).index("input_text")])

            if sample_table.iloc[row_idx[0], list(sample_table.columns).index("type")] == "multichoice" or \
                    sample_table.iloc[row_idx[0], list(sample_table.columns).index("type")] == "factual":
                unc_conf = list(sample_table.iloc[row_idx, list(sample_table.columns).index(
                    "unc_conf")])  # batches' confidence of one id
                unc_max_conf_idx = unc_conf.index(
                    max(unc_conf))  # index (in sample_table) of the batch with max confidence
                unc_pred = sample_table.iloc[
                    row_idx[unc_max_conf_idx], list(sample_table.columns).index("unc_pred")]

                con_conf = list(sample_table.iloc[row_idx, list(sample_table.columns).index(
                    "con_conf")])  # batches' confidence of one id
                con_max_conf_idx = con_conf.index(
                    max(con_conf))  # index (in sample_table) of the batch with max confidence
                con_pred = sample_table.iloc[
                    row_idx[con_max_conf_idx], list(sample_table.columns).index("con_pred")]

            elif sample_table.iloc[row_idx[0], list(sample_table.columns).index("type")] == "free-text":
                conf = list(sample_table.iloc[row_idx, list(sample_table.columns).index(
                    "unc_conf")])  # batches' confidence of one id
                if 1 in conf:
                    max_conf_idx = conf.index(max(conf))  # index (in sample_table) of the batch with max confidence
                    pred = sample_table.iloc[row_idx[max_conf_idx], list(sample_table.columns).index("unc_pred")]
                else:
                    c = np.array(conf.copy())
                    res = np.where(c >= conf_theshold, True, False)
                    if res.any():
                        pred = ""
                        inputs = ""
                        for c in np.arange(len(row_idx)):
                            if (conf[c] >= conf_theshold): pred = pred + sample_table.iloc[
                                row_idx[c], list(sample_table.columns).index("unc_pred")]
                    else:
                        max_conf_idx = conf.index(
                            max(conf))  # index (in sample_table) of the batch with max confidence
                        pred = sample_table.iloc[
                            row_idx[max_conf_idx], list(sample_table.columns).index("unc_pred")]

                pred = self.pp_final_pred(pred)
                unc_pred = pred
                con_pred = pred

            list_pred_unc.append(unc_pred)
            list_pred_con.append(con_pred)

        final_sample_table = pd.DataFrame(
            zip(unique_ids, list_type, list_input, list_ref, list_pred_unc, list_pred_con),
            columns=["id", "type", "input_text", "ref", "best_pred_unc", "best_pred_con"])
        final_sample_table.to_csv(self.res_dir + "EPC_" + tag + "_final_table_" + str(self.k) + ".csv")
        print("EPC_" + tag + "_final_table_" + str(self.k) + "fold: SAVED!")
        # display(final_sample_table.loc[final_sample_table["type"]=="free-text"])

        unc_sample_table = final_sample_table.iloc[:, 0:5].copy()
        unc_sample_table = unc_sample_table.rename(columns={"best_pred_unc": "best_pred"})
        unc_results = calculate_metrics_batch(unc_sample_table, self.tokenizer, self.model)
        for r in unc_results["metric"]:
            unc_results["metric"][r]["fold"] = self.k
        unc_df = pd.DataFrame(unc_results["metric"])
        unc_df.to_csv(self.res_dir + "EPC_" + tag + "_unc_results_" + str(self.k) + ".csv")
        print("EPC_" + tag + "_unc_results" + str(self.k) + "fold: SAVED!")
        print(unc_df)

        con_sample_table = final_sample_table.iloc[:, 0:4].copy()
        con_sample_table["best_pred_con"] = final_sample_table.iloc[:, 5].copy()
        con_sample_table = con_sample_table.rename(columns={"best_pred_con": "best_pred"})
        con_results = calculate_metrics_batch(con_sample_table, self.tokenizer, self.model)
        for r in con_results["metric"]:
            con_results["metric"][r]["fold"] = self.k
        con_df = pd.DataFrame(con_results["metric"])
        con_df.to_csv(self.res_dir + "EPC_"+tag+"_con_results_" + str(self.k) +".csv")
        print("EPC_" + tag + "_con_results" + str(self.k) + "fold: SAVED!")
        print(con_df)
        print("> EPC_pretrained_model (fold", self.k, "): REASSEMBLY OF BATCHES END")

class ExPostCombination_FinetunedModel:
    def __init__(self, k_fold=0, seed=config.SEEDS[0]):
        self.k = k_fold
        self.seed = seed
        self.output_dir = config.OUTPUT_DIR + "ex-post-comb/"
        self.ft_model = AutoModelForSeq2SeqLM.from_pretrained(self.output_dir + "output_" + str(self.k)+ "_eval-loss")
        self.ft_tokenizer = AutoTokenizer.from_pretrained(self.output_dir + "output_" + str(self.k)+ "_eval-loss")
        config.USE_PRETRAINED = False
        config.POST_PROCESSING = True
        self.res_dir = config.RESULTS_DIR + "ex-post-comb/finetuned/"
        # DATASET PREPARATION
        self.multitask_dataset = epc_k_fold_splitting(self.k, self.seed, self.ft_tokenizer)
        self.sample_test = self.multitask_dataset["test"]

    # Post-processing: free-text batch answers
    def pp_batch_pred(self,pred, ref):
        new_pred = []
        new_conf = []
        for i in np.arange(len(pred)):
            t = pred[i]
            resultantList = []
            for element in t.split(","):
                if element.strip() not in resultantList:
                    resultantList.append(element.strip())
            res = ""
            for n_el in np.arange(len(resultantList)):
                if n_el < len(resultantList) - 1:
                    # print(resultantList[n_el])
                    if resultantList[n_el] == "[not applicable]":
                        ""
                    else:
                        res = res + resultantList[n_el] + ", "
                else:
                    words = resultantList[n_el].split(" ")
                    es = re.search("[q|w|r|t|y|p|s|d|f|g|h|j|k|l|z|x|c|v|b|n|m|-]{1}$", words[-1])
                    if es: words.pop()
                    for v in np.arange(2):
                        if len(words) > 1 and len(words[-1]) <= 4:
                            words.pop()
                    if len(words) > 1 or (len(words) == 1 and len(words[0]) > 10):
                        for n_w in np.arange(len(words)):
                            res = res + words[n_w] + " "
            res = res.strip()
            conf = compute_f1(res, ref[i], self.ft_tokenizer)
            new_pred.append(res)
            new_conf.append(conf)
        return new_pred, new_conf

    # Post-processing: free-text final answers
    def pp_final_pred(self,pred):
        new_pred = []
        resultantList = []
        for element in pred.split(","):
            if element.strip() not in resultantList:
                resultantList.append(element.strip())
        res = ""
        for n_el in np.arange(len(resultantList)):
            if n_el < len(resultantList) - 1:
                res = res + resultantList[n_el] + ", "
            else:
                res = res + resultantList[n_el] + " "
        res = res.strip()
        if res[-1] == ",": res = res[:-1]
        return res

    # Loading text-generation hyperparameters: manually selecting the average values
    def param_text_gen(self):
        best_params = []
        for k in np.arange(config.K_FOLD):
            param_grid = pd.read_csv(self.res_dir + "EPC_param_table_" + str(k) + ".csv")
            max_values = param_grid.loc[param_grid["total_score"] == max(param_grid["total_score"])]
            best_params.append(list(max_values.iloc[0, 1:]))
        param_table = pd.DataFrame(best_params,
                                   columns=["repetition_penalty", "num_beams", "length_penalty", "conf_threshold",
                                            "strict_acc", "f1", "total_score"])

        #print(param_table)
        pm = param_table.mean()
        #print("mean:",pm)
        rp = 1.2
        nb = 2
        lp = 3
        conf = 0.2
        return [rp, nb, lp], conf

    # First validation: batches validation and post-processing
    def first_validation(self):
        print("> EPC_finetuned_model (fold", self.k, "): FIRST VALIDATION START")
        params, conf = self.param_text_gen()
        unc_results = validate(self.ft_model, self.ft_tokenizer, self.sample_test, constrained=False, opt=params)
        for r in unc_results["metric"]:
            unc_results["metric"][r]["fold"] = self.k
        unc_df = pd.DataFrame(unc_results["metric"])
        print(unc_df)
        unc_df.to_csv(self.res_dir + "EPC_FT_unc_results_batches_" + str(self.k) + ".csv")
        print(f"EPC_FT_unc_results {self.k} fold: SAVED!")

        con_results = validate(self.ft_model, self.ft_tokenizer, self.sample_test, constrained=True, opt=params)
        for r in con_results["metric"]:
            con_results["metric"][r]["fold"] = self.k
        con_df = pd.DataFrame(con_results["metric"])
        print(con_df)
        con_df.to_csv(self.res_dir + "EPC_FT_con_results_batches_" + str(self.k) + ".csv")
        print(f"EPC_FT_con_results {self.k} fold: SAVED!")

        prediction_table = pd.DataFrame({"id": self.sample_test["id"],
                                        "batches": self.sample_test["batches"],
                                        "type": self.sample_test["type"],
                                        "input_text": self.sample_test["input_text"],
                                        "ref": unc_results["references"],
                                        "unc_pred": unc_results["predictions"],
                                        "con_pred": con_results["predictions"],
                                        "unc_conf": unc_results["confidences"],
                                        "con_conf": con_results["confidences"]})
        print(prediction_table)
        prediction_table.to_csv(self.res_dir + "EPC_FT_table_batches_" + str(self.k) + ".csv")
        print(f"EPC_FT_table {self.k} fold: SAVED!")
        # post-processing
        ft_all = prediction_table.loc[prediction_table["type"] == "free-text"]
        b = ft_all.loc[prediction_table["ref"] != "[not applicable]"]
        tab = b.reset_index()
        new_pred, new_conf = self.pp_batch_pred(tab["unc_pred"], tab["ref"])
        print(new_pred)
        idx = tab["index"]
        new_tab = prediction_table.copy()

        for i in np.arange(len(idx)):
            new_tab.iloc[idx[i], 5] = new_pred[i]
            new_tab.iloc[idx[i], 6] = new_pred[i]
            new_tab.iloc[idx[i], 7] = new_conf[i]
            new_tab.iloc[idx[i], 8] = new_conf[i]
        new_tab.to_csv(self.res_dir + "PP_EPC_FT_table_batches_" + str(self.k) + ".csv")
        print(f"PP_EPC_FT_table {self.k} fold: SAVED!")
        print("> EPC_finetuned_model (fold", self.k, "): FIRST VALIDATION END")

    # Final validation:
    # - reassembly of batches
    # - selection/concatenation of best answer(s)
    # - post-processing
    # - calculation of metrics
    def reassembly_batches(self):
        param_grid = pd.read_csv(self.res_dir + "EPC_param_table_" + str(self.k) + ".csv")
        print(param_grid.loc[param_grid["total_score"] == max(param_grid["total_score"])])
        tag = "PP_FT"
        max_values = param_grid.loc[param_grid["total_score"] == max(param_grid["total_score"])]
        conf_theshold = 0.2
        sample_table = pd.read_csv(self.res_dir + "PP_EPC_FT_table_batches_" + str(self.k) + ".csv")

        unique_ids = list(sample_table['id'].unique())
        list_type = []
        list_ref = []
        list_input = []
        list_pred_unc = []
        list_pred_con = []

        for id in np.arange(len(unique_ids)):
            row_idx = list(sample_table.index[sample_table['id'] == unique_ids[id]])
            list_type.append(sample_table.iloc[row_idx[0], list(sample_table.columns).index("type")])
            list_ref.append(sample_table.iloc[row_idx[0], list(sample_table.columns).index("ref")])
            list_input.append(sample_table.iloc[row_idx[0], list(sample_table.columns).index("input_text")])

            if sample_table.iloc[row_idx[0], list(sample_table.columns).index("type")] == "multichoice" or \
                    sample_table.iloc[row_idx[0], list(sample_table.columns).index("type")] == "factual":
                unc_conf = list(sample_table.iloc[row_idx, list(sample_table.columns).index(
                    "unc_conf")])  # batches' confidence of one id
                unc_max_conf_idx = unc_conf.index(
                    max(unc_conf))  # index (in sample_table) of the batch with max confidence
                unc_pred = sample_table.iloc[
                    row_idx[unc_max_conf_idx], list(sample_table.columns).index("unc_pred")]

                con_conf = list(sample_table.iloc[row_idx, list(sample_table.columns).index(
                    "con_conf")])  # batches' confidence of one id
                con_max_conf_idx = con_conf.index(
                    max(con_conf))  # index (in sample_table) of the batch with max confidence
                con_pred = sample_table.iloc[
                    row_idx[con_max_conf_idx], list(sample_table.columns).index("con_pred")]

            elif sample_table.iloc[row_idx[0], list(sample_table.columns).index("type")] == "free-text":
                conf = list(sample_table.iloc[row_idx, list(sample_table.columns).index(
                    "unc_conf")])  # batches' confidence of one id
                if 1 in conf:
                    max_conf_idx = conf.index(max(conf))  # index (in sample_table) of the batch with max confidence
                    pred = sample_table.iloc[row_idx[max_conf_idx], list(sample_table.columns).index("unc_pred")]
                else:
                    c = np.array(conf.copy())
                    res = np.where(c >= conf_theshold, True, False)
                    if res.any():
                        pred = ""
                        inputs = ""
                        for c in np.arange(len(row_idx)):
                            if (conf[c] >= conf_theshold): pred = pred + sample_table.iloc[
                                row_idx[c], list(sample_table.columns).index("unc_pred")]
                    else:
                        max_conf_idx = conf.index(
                            max(conf))  # index (in sample_table) of the batch with max confidence
                        pred = sample_table.iloc[
                            row_idx[max_conf_idx], list(sample_table.columns).index("unc_pred")]


                pred = self.pp_final_pred(pred)
                unc_pred = pred
                con_pred = pred

            list_pred_unc.append(unc_pred)
            list_pred_con.append(con_pred)

        final_sample_table = pd.DataFrame(
            zip(unique_ids, list_type, list_input, list_ref, list_pred_unc, list_pred_con),
            columns=["id", "type", "input_text", "ref", "best_pred_unc", "best_pred_con"])
        final_sample_table.to_csv(self.res_dir + "EPC_" + tag + "_final_table_" + str(self.k) + ".csv")
        print("EPC_" + tag + "_final_table_" + str(self.k) + "fold: SAVED!")
        # display(final_sample_table.loc[final_sample_table["type"]=="free-text"])

        unc_sample_table = final_sample_table.iloc[:, 0:5].copy()
        unc_sample_table = unc_sample_table.rename(columns={"best_pred_unc": "best_pred"})
        unc_results = calculate_metrics_batch(unc_sample_table, self.ft_tokenizer, self.ft_model)

        con_sample_table = final_sample_table.iloc[:, 0:4].copy()
        con_sample_table["best_pred_con"] = final_sample_table.iloc[:, 5].copy()
        con_sample_table = con_sample_table.rename(columns={"best_pred_con": "best_pred"})
        con_results = calculate_metrics_batch(con_sample_table, self.ft_tokenizer, self.ft_model)

        for r in unc_results["metric"]:
            unc_results["metric"][r]["fold"] = self.k
        unc_df = pd.DataFrame(unc_results["metric"])

        for r in con_results["metric"]:
            con_results["metric"][r]["fold"] = self.k
        con_df = pd.DataFrame(con_results["metric"])

        # Final metrics
        unc_df.to_csv(self.res_dir + "EPC_"+tag+"_unc_results_" + str(self.k) +".csv")
        print("EPC_" + tag + "_unc_results" + str(self.k) + "fold: SAVED!")
        print(unc_df)
        con_df.to_csv(self.res_dir + "EPC_"+tag+"_con_results_" + str(self.k) +".csv")
        print("EPC_" + tag + "_con_results" + str(self.k) + "fold: SAVED!")
        print(con_df)
