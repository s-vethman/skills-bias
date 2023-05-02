## Import Initialization

# standard packages
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
# packages for statistical analysis
from statsmodels.stats.weightstats import ttest_ind
import scipy.stats as stats
from bioinfokit.analys import stat
import statsmodels.api as sm
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times New Roman']

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
    fig.set_dpi(300)
    fig.suptitle('')
    csfont = {'fontname': 'Times New Roman'}
    hfont = {'fontname': 'Times New Roman'}
    plt.ylim(-4, 4)
    plt.xlabel("Male employment ratio(%)", **hfont)
    plt.ylabel("Skill WEAT score", **hfont)
    plt.title(title_name, **csfont)

    plt.savefig(figure_output_path)

def get_boxplots(dfE,dfO, figure_output_path):
    dfE['C'] = dfE.MalePercentage
    dfE['C'] = round(np.ceil(dfE['C'] / 25) * 25)
    dfE['C'] = dfE['C'].replace([25, 50, 75, 100], ["0-25", "25-75", "25-75", "75-100"])

    dfO['C'] = dfO.MalePercentage
    dfO['C'] = round(np.ceil(dfO['C'] / 25) * 25)
    dfO['C'] = dfO['C'].replace([25, 50, 75, 100], ["0-25", "25-75", "25-75", "75-100"])

    fig, ((ax1, ax2)) = plt.subplots(1, 2)

    dfO.boxplot(column='SkillWEATscores', by='C', whis=(5, 95), ax=ax1)
    dfE.boxplot(column='SkillWEATscores', by='C', whis=(5, 95), ax=ax2)

    ax1.title.set_text('O*NET')
    ax2.title.set_text('ESCO')

    ax1.set(xlabel="Male employment ratio(%)", ylabel="Skill WEAT score")
    ax2.set(xlabel="Male employment ratio(%)", ylabel="Skill WEAT score")
    fig.set_size_inches(7, 3.5)
    fig.set_dpi(300)
    fig.suptitle('')
    plt.ylim(-4, 4)
    # csfont = {'fontname': 'Times New Roman'}
    # hfont = {'fontname': 'Times New Roman'}
    # plt.xlabel("Male employment ratio(%)", **hfont)
    # plt.ylabel("Skill WEAT score", **hfont)
    # plt.title(title_name, **csfont)

    plt.savefig(figure_output_path)

def get_qqplot(dfE,dfO ,pathqq):
    dfE['C'] = dfE.MalePercentage
    dfE['C'] = round(np.ceil(dfE['C'] / 25) * 25)
    dfE['C'] = dfE['C'].replace([25, 50, 75, 100], ["0-25", "25-75", "25-75", "75-100"])

    dfO['C'] = dfO.MalePercentage
    dfO['C'] = round(np.ceil(dfO['C'] / 25) * 25)
    dfO['C'] = dfO['C'].replace([25, 50, 75, 100], ["0-25", "25-75", "25-75", "75-100"])

    fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3)
    sm.qqplot(dfO[dfO["C"]==  "0-25"]['SkillWEATscores'], line='s', ax = ax1)
    sm.qqplot(dfO[dfO["C"] == "25-75"]['SkillWEATscores'], line='s', ax = ax2)
    sm.qqplot(dfO[dfO["C"] == "75-100"]['SkillWEATscores'], line='s', ax = ax3)
    sm.qqplot(dfE[dfE["C"] == "0-25"]['SkillWEATscores'], line='s', ax=ax4)
    sm.qqplot(dfE[dfE["C"] == "25-75"]['SkillWEATscores'], line='s', ax=ax5)
    sm.qqplot(dfE[dfE["C"] == "75-100"]['SkillWEATscores'], line='s', ax=ax6)

    ax1.title.set_text('Female dominated occupations')
    ax2.title.set_text('Mixed occupations')
    ax2.set(ylabel=None)
    ax3.title.set_text('Male dominated occupations')
    ax3.set(ylabel=None)
    ax5.set(ylabel=None)
    ax6.set(ylabel=None)
    fig.set_dpi(300)
    fig.set_size_inches(8, 5)
    #fig.suptitle("QQ-plots to test Normal distributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(pathqq)


def get_analysis(df,database_name,path,pathqq):
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
def get_analysis_hist(dfE,dfO,path):

    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    sns.histplot(x=dfO.SkillWEATscores,  ax = ax1, color = "blue")
    sns.histplot(x=dfE.SkillWEATscores, ax=ax2, color = "blue")

    ax1.title.set_text('O*NET')
    ax2.title.set_text('ESCO')
    # hfont = {'fontname': 'Times New Roman'}
    ax1.set(ylabel="Frequency", xlabel="Skill WEAT score", xlim = (-4,4))
    ax2.set(ylabel="Frequency", xlabel="Skill WEAT score", xlim = (-4,4))
    fig.set_size_inches(7, 3)
    fig.set_dpi(300)
    fig.suptitle('')
    # plt.ylim(-1.75, 1.75)
    plt.tight_layout()
    plt.savefig(path)

def get_analysis_avg_plot(avgO,avgE,path):

    xO = avgO.MalePercentage
    yO = avgO.SkillWEATscores
    xE = avgE.MalePercentage
    yE = avgE.SkillWEATscores

    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    sns.regplot(x=xO, y=yO, ci=None, ax=ax1, scatter_kws={"color": "blue","s" : 13}, line_kws={"color": "black"})
    sns.regplot(x=xE, y=yE, ci=None, ax=ax2,  scatter_kws={"color": "blue","s": 13}, line_kws={"color": "black"})

    ax1.title.set_text('O*NET')
    ax2.title.set_text('ESCO')
    ax1.set(xlabel="Male employment ratio(%)", ylabel="Skill WEAT score", ylim = (-1.9, 1.9),xlim = (0, 100))
    ax2.set(xlabel="Male employment ratio(%)", ylabel="Skill WEAT score", ylim = (-1.9, 1.9),xlim = (0, 100))
    ax1.text(55,-1.5,"y = 0.0074x + 0.2986")
    ax2.text(55, -1.5, "y = 0.0074x - 0.2259")

    fig.set_size_inches(7.5, 3.5)
    fig.set_dpi(300)
    fig.suptitle('')
    plt.ylim(-1.75, 1.75)
    plt.tight_layout()
    plt.savefig(path)


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

    # Load the input from main_analysis()
    dfE = pd.read_csv("data/OUTPUT_ESCO_skills.csv", sep="\t")
    dfO = pd.read_csv("data/OUTPUT_ONET_skills.csv", sep="\t")
    job_averages_O = pd.read_csv("data/OUTPUT_ONET_skills_avg.csv", sep=",")
    job_averages_E = pd.read_csv("data/OUTPUT_ESCO_skills_avg.csv", sep=",")

    # Create boxplots and get correlations and significance scores
    get_analysis(dfO, "O*NET", "data/ONET_boxplot.JPEG","data/ONET_qqplot.JPEG")
    get_analysis_avg(job_averages_O,"O*NET")
    get_analysis(dfE, "ESCO", "data/ESCO_boxplot.JPEG","data/ESCO_qqplot.JPEG")
    get_analysis_avg(job_averages_E, "ESCO")

    # Get the figures used in the article
    get_analysis_hist(dfE, dfO, "images/fig1.pdf")
    get_boxplots(dfE, dfO, "images/fig2.pdf")
    get_analysis_avg_plot(job_averages_O,job_averages_E,"images/fig3.pdf")
    get_qqplot(dfE, dfO, "images/fig4.pdf")

if __name__ == "__main__":
    #choose which main you would like to run
    # main_analysis()
    main_results()


