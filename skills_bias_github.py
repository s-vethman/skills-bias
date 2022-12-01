## Import Initialization

# standard packages
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

# packages for statistical analysis
from statsmodels.stats.weightstats import ttest_ind
import scipy.stats as stats
from bioinfokit.analys import stat
import statsmodels.api as sm

# packages for BERT
from transformers import BertTokenizer, BertForMaskedLM

# global variables chosen for BERT analysis:
BERT_MODEL = 'bert-base-uncased'
TEMPLATE = "twosentences"  # "onesentence"


class BiasDetermination():
    def __init__(self):

        # Initialize the matcher with the shared vocab
        self._tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self._modelmask = BertForMaskedLM.from_pretrained(BERT_MODEL)

    def SkillWEAT(self, skillList):

        # Prior Probabilities: For all skills in the skill list, we will get the prior probabilities, aka the benchmark probabilities, once.
        preds, mask_index = self.MaskPredTask(
            skill=None, tokenizer=self._tokenizer, prior=True
        )
        priors = self.Pred2BiasProb(self._tokenizer, preds, mask_index)

        priorMan = np.array(priors.loc[priors['Predicted'] == 'he', 'Prob'])
        priorWoman = np.array(priors.loc[priors['Predicted'] == 'she', 'Prob'])

        # Set up of the data frame in which we will store the output
        WEAToutput = pd.DataFrame(columns=["Skill", "WEATscore"])
        # For every skill in the skill-list, we acquire the WEAT Score.

        for i, sk in enumerate(skillList):
            # First we go from skill to mask predictions relevant to WEAT.

            preds, mask_index = self.MaskPredTask(
                sk, self._tokenizer, prior=False
            )
            # Then we convert the prediction to probabilities and collect the relevant ones for WEAT.
            output = self.Pred2BiasProb(self._tokenizer, preds, mask_index)

            predMan = np.array(output.loc[output['Predicted'] == 'he', 'Prob'])
            predWoman = np.array(output.loc[output['Predicted'] == 'she', 'Prob'])

            # Calculate Weat Score from predictions and priors if the input was specified correctly
            weat_score = 0
            if (priorMan == 0 or priorWoman == 0):
                print(" Warning: Text does not fit into Template")
            else:
                log_bias_score = np.log(predMan / priorMan) - np.log(predWoman / priorWoman)
                weat_score = log_bias_score.item()

            WEAToutput = WEAToutput.append(
                {"Skill": sk, "WEATscore": weat_score},
                ignore_index=True,
            )

        return WEAToutput

    def MaskPredTask(self, skill, tokenizer, prior=False):
        """Go from skill provided, to text consisting of the WEAT template (with [MASK] included)
        Afterwards, text is converted (I)to tokenized text with sentence_ids and finally (II)evaluated in the model (III) resulting to prediction of the masked index."""

        # Skill is put in the the template chosen, dependent on the Language and TEMPLATE type of one or two sentences.
        # If prior is False we get template for mask prediction related to the skill
        # If prior is True we get template for mask prediction unrelated to a specific skill, that is the benchmark.
        # The benchmark allows you to see how the incorporation of the skill in the template affects the prediction.

        text = ""

        if TEMPLATE == "onesentence":

            if (prior == False):
                text = f"[CLS] [MASK] can {skill} . [SEP]"
            else:
                text = f"[CLS] [MASK] can [MASK] . [SEP] "

            # Tokenize input and get masked_index
            tokenized_text = tokenizer.tokenize(text)
            masked_index = tokenized_text.index('[MASK]')
            # Get the corresponding ids for sentence 1 and 2.
            segments_ids = [0] * len(tokenized_text)

        elif TEMPLATE == "twosentences":

            if (prior == False):
                text = f"[CLS] [MASK] can {skill} . [SEP] [MASK] is suitable for the job . [SEP]"
            else:
                text = f"[CLS] [MASK] can [MASK] . [SEP] [MASK] is suitable for the job . [SEP]"

            # print(text)
            # Tokenize input and get masked_index
            tokenized_text = tokenizer.tokenize(text)
            masked_index = tokenized_text.index('[MASK]')
            # Get the corresponding ids for sentence 1 and 2.
            sent1 = tokenized_text.index('[SEP]') + 1
            sent2 = len(tokenized_text)
            segments_ids = [0] * (sent1) + [1] * (sent2 - sent1)
        else:
            print(
                f" Template specified is wrong , change variable TEMPLATE to \"onesentence\" or \"twosentences\", now it is {TEMPLATE}.")

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Evaluate the model once and then make all relevant predictions.
        self._modelmask.eval()

        # Predict all tokens
        with torch.no_grad():
            outputs = self._modelmask(
                tokens_tensor, token_type_ids=segments_tensors
            )
            predictions = outputs[0]

        return predictions, masked_index

    def Pred2BiasProb(self, tokenizer, preds, mask_index, top_k=300):
        """From Masked Prediction output this function first creates top k predictions and their associated probabilities via the method softmax of torch.nn.
        Then from this output of predicted words and probabilities, collects the relevant bias output for the gender dimension. """
        # First from preds to top k probs
        probs = torch.nn.functional.softmax(preds[0, mask_index], dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
        # Collect the top k probs in a readable output with their predicted words
        output = pd.DataFrame(
            [
                tokenizer.convert_ids_to_tokens([pred_idx])[0],
                top_k_weights[i].item(),
            ]
            for i, pred_idx in enumerate(top_k_indices)
        )
        output.columns = ["Predicted", "Prob"]
        # Select the relavant predicted words for the bias in the gender dimension.
        biasOutput = output.loc[output["Predicted"].isin(["she", "he"])]

        return biasOutput

def get_stat_results(df):
    res = ttest_ind(df[df['C'] == '0-25']['SkillWEATscores'], df[df['C'] == '75-100']['SkillWEATscores'])
    print("t-test male versus female")
    print(res)
    fvalue, pvalue = stats.f_oneway(df[df['C'] == '0-25']['SkillWEATscores'], df[df['C'] == '25-75']['SkillWEATscores'],
                                    df[df['C'] == '75-100']['SkillWEATscores'])
    print("ANOVA: F-value and P-value")
    print(fvalue, pvalue)
    res = stat()
    print("Tukey HSD results")
    res.tukey_hsd(df=df, res_var='SkillWEATscores', xfac_var='C', anova_model='SkillWEATscores ~ C')
    print(res.tukey_summary)

def get_boxplot(df, title_name, figure_output_path):
    axes = df.boxplot(column='SkillWEATscores', by='C', whis=(5, 95))

    for tick in axes.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in axes.get_yticklabels():
        tick.set_fontname("Times New Roman")

    fig = axes.get_figure()
    fig.set_size_inches(4.5, 4.5)
    fig.set_dpi(100)
    fig.suptitle('')
    csfont = {'fontname': 'Times New Roman'}
    hfont = {'fontname': 'Times New Roman'}
    plt.ylim(-4, 4)
    plt.xlabel("Male employment ratio(%)", **hfont)
    plt.ylabel("Skill WEAT score", **hfont)
    plt.title(title_name, **csfont)

    plt.savefig(figure_output_path)


def get_analysis(df,database_name,path):
    df['C'] = df.MalePercentage
    df['C'] = round(np.ceil(df['C'] / 25) * 25)
    df.head()
    df['C'] = df['C'].replace([25, 50, 75, 100], ["0-25", "25-75", "25-75", "75-100"])

    get_boxplot(df,database_name,path)
    get_stat_results(df)

def get_analysis_avg(avg,database_name):

    x = avg.MalePercentage
    y = avg.SkillWEATscores

    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    print("----"+database_name+"----")
    print(results.summary())
    print("The correlation is")
    print(np.sqrt(results.rsquared))


def save_df(df_,fn,fn_avg):
    df_.to_csv(fn, index=False, header=True, sep='\t')
    job_averages_ = df_.groupby('JobTitle').mean()
    job_averages_.to_csv(fn_avg, header=True, sep=',')


# you can use main to test the code
def main_analysis():
    # Initialize the BERT model for both ONET AND ESCO
    biasDeterminer = BiasDetermination()

    # ONET
    INPUT_FILENAME = "data/ONET_Skills.txt"
    OUTPUT_FILENAME = "data/OUTPUT_ONET_skills.csv"
    OUTPUT_FILENAME_avg = "data/OUTPUT_ONET_skills_avg.csv"
    df = pd.read_csv(INPUT_FILENAME, sep="\t")
    df['MalePercentage'] = df['MalePercentage'].str.replace(',', '.').astype(float)
    scores = biasDeterminer.SkillWEAT(df.Skill)
    df['SkillWEATscores'] = scores['WEATscore']
    save_df(df, OUTPUT_FILENAME, OUTPUT_FILENAME_avg)


    # ESCO
    INPUT_FILENAME = "data/ESCO_Skills.txt"
    OUTPUT_FILENAME = "data/OUTPUT_ESCO_skills.csv"
    OUTPUT_FILENAME_avg = "data/OUTPUT_ESCO_skills_avg.csv"
    df = pd.read_csv(INPUT_FILENAME, sep="\t")
    df['MalePercentage'] = df['MalePercentage'].str.replace(',', '.').astype(float)
    scores = biasDeterminer.SkillWEAT(df.Skill)
    df['SkillWEATscores'] = scores['WEATscore']
    save_df(df,OUTPUT_FILENAME,OUTPUT_FILENAME_avg)


def main_results():

    # ONET
    OUTPUT_FILENAME = "data/OUTPUT_ONET_skills.csv"
    OUTPUT_FILENAME_avg = "data/OUTPUT_ONET_skills_avg.csv"
    # Load the input from main_analysis()
    df = pd.read_csv(OUTPUT_FILENAME, sep="\t")
    job_averages = pd.read_csv(OUTPUT_FILENAME_avg, sep=",")
    # create boxplots and get significance scores
    get_analysis(df, "O*NET", "data/ONET_boxplot.png")
    get_analysis_avg(job_averages,"O*NET")

    # ESCO
    OUTPUT_FILENAME = "data/OUTPUT_ESCO_skills.csv"
    OUTPUT_FILENAME_avg = "data/OUTPUT_ESCO_skills_avg.csv"
    # Load the input from main_analysis()
    df = pd.read_csv(OUTPUT_FILENAME, sep="\t")
    job_averages = pd.read_csv(OUTPUT_FILENAME_avg, sep=",")
    # create boxplots and get correlations and significance scores
    get_analysis(df, "ESCO", "data/ESCO_boxplot.png")
    get_analysis_avg(job_averages, "ESCO")

if __name__ == "__main__":
    #choose which main you would like to run
    #main_analysis()
    main_results()


