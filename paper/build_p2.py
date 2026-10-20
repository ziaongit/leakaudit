"""Build reframed Paper 2 (leakage audit) as .docx. Tables are filled from the result JSON files."""
import json, re, pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

R = {}
for f in ['A', 'Anocharge', 'B', 'rule5', 'lime']: R.update(json.load(open(f'results/audit_{f}.json')))
B_ONLY = dict(f1=0.614, sd=0.035, auc=0.924, ap=0.659)      # b_single.py, LightGBM, 5 splits
B_REM = dict(f1=0.654, sd=0.027, auc=0.936, ap=0.711)

REFS = {
 'hhs': 'U.S. Department of Health and Human Services: Medicare and Medicaid Fraud, Waste, and Abuse: Overview. Washington, DC (2023)',
 'bauder17': 'Bauder, R.A., Khoshgoftaar, T.M.: Medicare fraud detection using machine learning methods. In: 2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA), pp. 858–865. IEEE (2017). https://doi.org/10.1109/ICMLA.2017.00-48',
 'johnson19': 'Johnson, J.M., Khoshgoftaar, T.M.: Medicare fraud detection using neural networks. J. Big Data 6, 63 (2019). https://doi.org/10.1186/s40537-019-0225-0',
 'razzaq': 'Razzaq, K., Shah, M.: Next-generation machine learning in healthcare fraud detection: current trends, challenges, and future research directions. Information 16(9), 730 (2025). https://doi.org/10.3390/info16090730',
 'nabrawi': 'Nabrawi, E., Alanazi, A.: Fraud detection in healthcare insurance claims using machine learning. Risks 11(9), 160 (2023). https://doi.org/10.3390/risks11090160',
 'aruleba': 'Aruleba, I.T., Sun, Y.: HealthCare fraud detection using machine learning. Preprint, not peer reviewed. SSRN (2023). https://ssrn.com/abstract=4631193',
 'wang': 'Wang, Z., Chen, X., Wu, Y., Jiang, L., Lin, S., Qiu, G.: A robust and interpretable ensemble machine learning model for predicting healthcare insurance fraud. Sci. Rep. 15, 218 (2025). https://doi.org/10.1038/s41598-024-82062-x',
 'kaufman': 'Kaufman, S., Rosset, S., Perlich, C., Stitelman, O.: Leakage in data mining: formulation, detection, and avoidance. ACM Trans. Knowl. Discov. Data 6(4), 15 (2012). https://doi.org/10.1145/2382577.2382579',
 'kapoor': 'Kapoor, S., Narayanan, A.: Leakage and the reproducibility crisis in machine-learning-based science. Patterns 4(9), 100804 (2023). https://doi.org/10.1016/j.patter.2023.100804',
 'lundberg17': 'Lundberg, S.M., Lee, S.-I.: A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems 30, pp. 4765–4774 (2017)',
 'ribeiro': 'Ribeiro, M.T., Singh, S., Guestrin, C.: “Why should I trust you?”: explaining the predictions of any classifier. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1135–1144. ACM (2016). https://doi.org/10.1145/2939672.2939778',
 'manyike': 'Manyike, G., Paepae, T.: Enhancing healthcare insurance fraud detection using explainable ensemble machine learning on imbalanced datasets. Sci. Rep. (2026). https://doi.org/10.1038/s41598-026-66566-2',
 'lapuschkin': 'Lapuschkin, S., Wäldchen, S., Binder, A., Montavon, G., Samek, W., Müller, K.-R.: Unmasking Clever Hans predictors and assessing what machines really learn. Nat. Commun. 10, 1096 (2019). https://doi.org/10.1038/s41467-019-08987-4',
 'degrave': 'DeGrave, A.J., Janizek, J.D., Lee, S.-I.: AI for radiographic COVID-19 detection selects shortcuts over signal. Nat. Mach. Intell. 3, 610–619 (2021). https://doi.org/10.1038/s42256-021-00338-7',
 'thornton': 'Thornton, D., Mueller, R.M., Schoutsen, P., van Hillegersberg, J.: Predicting healthcare fraud in Medicaid: a multidimensional data model and analysis techniques for fraud detection. Procedia Technol. 9, 1252–1264 (2013). https://doi.org/10.1016/j.protcy.2013.12.140',
 'bauder18': 'Bauder, R.A., Khoshgoftaar, T.M.: The effects of varying class distribution on learner behavior for Medicare fraud detection with imbalanced big data. Health Inf. Sci. Syst. 6, 9 (2018). https://doi.org/10.1007/s13755-018-0051-3',
 'shekhar': 'Shekhar, S., Leder-Luis, J., Akoglu, L.: Can machine learning target health care fraud? Evidence from Medicare hospitalizations. J. Policy Anal. Manag. 45(1), e70078 (2026). https://doi.org/10.1002/pam.70078',
 'hancock': 'Hancock, J.T., Bauder, R.A., Wang, H., Khoshgoftaar, T.M.: Explainable machine learning models for Medicare fraud detection. J. Big Data 10, 154 (2023). https://doi.org/10.1186/s40537-023-00821-5',
 'payrov': 'Payrovnaziri, S.N., Chen, Z., Rengifo-Moreno, P., Miller, T., Bian, J., Chen, J.H., Liu, X., He, Z.: Explainable artificial intelligence models using real-world electronic health record data: a systematic scoping review. J. Am. Med. Inform. Assoc. 27(7), 1173–1185 (2020). https://doi.org/10.1093/jamia/ocaa053',
 'barredo': 'Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., García, S., Gil-López, S., Molina, D., Benjamins, R., Chatila, R., Herrera, F.: Explainable artificial intelligence (XAI): concepts, taxonomies, opportunities and challenges toward responsible AI. Inf. Fusion 58, 82–115 (2020). https://doi.org/10.1016/j.inffus.2019.12.012',
 'geirhos': 'Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F.A.: Shortcut learning in deep neural networks. Nat. Mach. Intell. 2, 665–673 (2020). https://doi.org/10.1038/s42256-020-00257-z',
 'jethani': 'Jethani, N., Saporta, A., Ranganath, R.: Don’t be fooled: label leakage in explanation methods and the importance of their quantitative evaluation. In: Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 206 (2023)',
 'alvarez': 'Alvarez-Melis, D., Jaakkola, T.S.: On the robustness of interpretability methods. arXiv:1806.08049 (2018)',
 'salih': 'Salih, A.M., Raisi-Estabragh, Z., Galazzo, I.B., Radeva, P., Petersen, S.E., Lekadir, K., Menegaz, G.: A perspective on explainable artificial intelligence methods: SHAP and LIME. Adv. Intell. Syst. 7(1), 2400304 (2025). https://doi.org/10.1002/aisy.202400304',
 'tamil': 'Tamilselvan, K.: Healthcare providers data. Kaggle. https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data (2021). Accessed [date]',
 'gupta': 'Gupta, R.A.: Healthcare provider fraud detection analysis. Kaggle. https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis. Accessed [date]',
 'ke': 'Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.Y.: LightGBM: a highly efficient gradient boosting decision tree. In: Advances in Neural Information Processing Systems 30, pp. 3146–3154 (2017)',
 'chen': 'Chen, T., Guestrin, C.: XGBoost: a scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794. ACM (2016). https://doi.org/10.1145/2939672.2939785',
 'breiman': 'Breiman, L.: Random forests. Mach. Learn. 45, 5–32 (2001). https://doi.org/10.1023/A:1010933404324',
 'lundberg20': 'Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., Lee, S.-I.: From local explanations to global understanding with explainable AI for trees. Nat. Mach. Intell. 2, 56–67 (2020). https://doi.org/10.1038/s42256-019-0138-9',
}

# ---------------- content ----------------
TITLE = 'Is Your Fraud Model Learning Fraud or the Labelling Rule? An Explanation-Based Leakage Audit with SHAP and LIME'
AUTH = 'Zia Ullah*, Wahab Khan'
AFF = 'Department of Computer Science, City University of Science and Information Technology (CUSIT), Peshawar, Pakistan'
CORR = '* Corresponding author: Zia Ullah, zia.ullah@valueadd.se'

ABS = [
 ('Purpose: ', 'A fraud model can reach an F1 of 0.96 on a public Medicare dataset using only two features, because those two features define the label. Many studies on public Medicare datasets report accuracy above 90%, but when labels are derived from the input features, such scores may reflect a relearned labelling rule rather than real fraud detection. This study tests whether SHAP and LIME can expose this problem.'),
 ('Methods: ', 'We propose a five-step audit that combines explanation methods with simple retraining checks: attribution concentration with SHAP, a model trained only on the suspect features, removal of the suspect features with a check for stand-in features, stability of global explanations across splits and models, and repeated LIME explanations of single records. The audit was applied with LightGBM, XGBoost and Random Forest over five splits to two datasets: dataset A (100,000 records), with a z-score label on submitted charges within each provider type, and dataset B (5,410 providers), with labels supplied with the data.'),
 ('Results: ', 'On dataset A, Average Submitted Charge Amount ranked first in every run. Models using only charge and provider type matched the full models (F1 0.947–0.960). Removing these features cut F1 to 0.495–0.530, and a related payment feature took over. Explanations were highly stable on dataset A but much less so on dataset B (Spearman ρ across models 0.893 vs 0.445), because all models learned the same rule. On dataset B, the top feature alone performed below the full model, and removing it caused no loss.'),
 ('Conclusion: ', 'Explanation methods can reveal label leakage, but only when combined with simple retraining tests. High explanation stability should not be read as evidence of a valid model.'),
]
KEYWORDS = 'Explainable AI · SHAP · LIME · Data leakage · Healthcare fraud detection · Medicare'

S = []   # list of (kind, payload)
def H1(t): S.append(('h1', t))
def H2(t): S.append(('h2', t))
def P(t): S.append(('p', t))
def BUL(items, num=False): S.append(('list', (items, num)))
def TAB(n): S.append(('table', n))
def FIG(path, cap): S.append(('fig', (path, cap)))

H1('1 Introduction')
P('Healthcare fraud costs public insurance programmes and patients money and erodes trust in the system {hhs}. Machine learning is increasingly used to decide which providers to review, and public datasets built from Medicare data have become the standard way to develop and compare fraud models {bauder17,johnson19,razzaq}. Many published models on these datasets report very high accuracy, often above 90% {nabrawi,aruleba,wang}. Such numbers are only meaningful if the labels reflect fraud and the model learned something beyond the way the labels were made.')
P('That condition is often not checked. Confirmed fraud labels are scarce, so many public datasets come with labels of unclear origin or with no labels at all. Users then build a label themselves, for example by flagging providers whose charges are unusually high for their speciality. A label built this way is a fixed function of some input features. Any flexible model given those features can rebuild the rule almost perfectly. The resulting scores measure how well the rule is relearned, not how well fraud is detected. This is a form of data leakage {kaufman}, and leakage has been identified as a leading cause of irreproducible results in ML-based research {kapoor}.')
P('Explainable AI methods such as SHAP {lundberg17} and LIME {ribeiro} are widely used in fraud detection to show investigators why a provider was flagged {wang,manyike}. In other fields, explanations have also exposed models that rely on shortcuts rather than the intended signal {lapuschkin,degrave}. This suggests a second use for them in fraud research: not only explaining a model’s decisions, but checking whether the data allow a valid model at all.')
P('This paper tests that idea. We ask three questions:')
BUL(['RQ1. Can SHAP and LIME reveal that a fraud label was derived from the input features?',
     'RQ2. Which simple checks are needed to turn an explanation-based suspicion into firm evidence of leakage?',
     'RQ3. How do the explanations behave on a dataset whose labels were not derived from its features?'])
P('To answer them, we study two public Medicare datasets. Dataset A is the public Healthcare Providers dataset, with labels assigned by a z-score rule on submitted charges within each provider type. Dataset B is a provider fraud dataset whose labels come with the data and are not computed from its features. On dataset A, SHAP placed the submitted charge first in every run. A model trained on only the two features that define the label matched the full model (F1 0.947–0.960). Removing these features cut F1 to 0.495–0.530, and a closely related payment feature took over as the top-ranked feature. Dataset B showed none of these signs: its top feature alone performed below the full model, and removing it caused no loss.')
P('This paper makes four contributions:')
BUL(['A five-step audit that combines SHAP and LIME explanations with suspect-only retraining, removal with a check for stand-in features, and stability measures. It needs only a trained model and its data.',
     'A case study on a public Kaggle dataset, showing that strong reported performance on dataset A mainly reflects the labelling rule.',
     'Evidence that explanation stability can mislead: explanations were most stable on the leaky dataset, because every model learned the same rule.',
     'A comparison with supplied labels, showing that a dominant feature alone does not prove leakage, and that the suspect-only test separates the two cases.'], num=True)
P('The rest of the paper is organised as follows. Section 2 reviews related work. Section 3 describes the datasets. Section 4 presents the audit procedure. Section 5 reports the results, and Section 6 discusses what they mean. Section 7 lists the limitations, and Section 8 concludes.')

H1('2 Related Work')
H2('2.1 Machine learning for healthcare fraud and the source of labels')
P('Machine learning for healthcare fraud detection has grown with the release of public claims and payment data. Thornton et al. {thornton} proposed a multidimensional data model for Medicaid fraud. Bauder and Khoshgoftaar {bauder17,bauder18} and Johnson and Khoshgoftaar {johnson19} trained classifiers on CMS Medicare data, and studied class imbalance and neural networks at scale. More recent studies report strong results with ensembles and hybrid models {nabrawi,aruleba,wang,manyike}. A recent survey notes that Medicare, LEIE and Kaggle datasets have become common benchmarks {razzaq}.')
P('These benchmarks differ sharply in where their labels come from. In the Bauder and Khoshgoftaar line of work, labels come from the List of Excluded Individuals and Entities (LEIE), which records providers excluded from federal programmes. These labels are external to the claims data, although they are incomplete. Hancock et al. {hancock} built explainable Medicare fraud models on LEIE-labelled data. Shekhar et al. {shekhar} took a similar external route by validating an unsupervised method against hospitals facing Department of Justice fraud lawsuits. Many Kaggle datasets, in contrast, either come with labels whose origin is not documented or have no labels, so users define one themselves. When a dataset has no label, as with the Healthcare Providers dataset studied here, the user must define one, for example with a statistical outlier rule on billing amounts. Studies that report scores on such data rarely state how the label relates to the input features. That relationship decides what a high score actually means.')
H2('2.2 Explainable AI in fraud detection')
P('SHAP {lundberg17} and LIME {ribeiro} are the two most widely used post-hoc explanation methods. SHAP assigns each feature a contribution based on Shapley values. LIME fits a simple local model around a single prediction. Both have been applied to healthcare fraud models to show investigators why a provider was flagged {wang,manyike}, and reviews of explainable AI in health stress their value for trust and accountability {payrov,barredo}. In nearly all of this work, explanations are used to justify a model’s output. They are rarely used to test whether the data and labels support the model in the first place.')
H2('2.3 Leakage and shortcut learning')
P('Data leakage occurs when information that would not be available at prediction time, or that directly encodes the target, enters the training data. Kaufman et al. {kaufman} formalised leakage in data mining and showed how it inflates reported performance. Kapoor and Narayanan {kapoor} reviewed published ML-based science across many fields and found leakage to be a leading cause of results that do not reproduce. A related problem is shortcut learning: a model solves the task through a simple cue that does not reflect the intended concept {geirhos}.')
P('Explanations have played a key role in finding such shortcuts. Lapuschkin et al. {lapuschkin} used relevance maps to expose “Clever Hans” classifiers that relied on watermarks and other artefacts. DeGrave et al. {degrave} used saliency methods to show that COVID-19 chest X-ray models relied on dataset-specific markers rather than lung findings. These studies are mostly about images. To our knowledge, explanation-based checks for leakage have not been applied systematically to tabular healthcare fraud data, where the label rule is often unstated. The term label leakage is also used for leakage inside explanation methods themselves {jethani}; here we mean leakage from the label definition into the input features.')
H2('2.4 Reliability of explanations')
P('Explanations have known weaknesses. Alvarez-Melis and Jaakkola {alvarez} showed that popular methods, LIME among them, can give quite different explanations for nearly identical inputs. Salih et al. {salih} found that SHAP and LIME outputs depend on the model and are affected by correlated features. These findings have led to stability being proposed as a quality measure: an explanation that stays the same across runs and models is treated as more trustworthy. Our results question this when it is used on its own. Explanations can be very stable precisely because every model has learned the same label rule.')
H2('2.5 Gap addressed by this study')
P('Earlier work either uses explanations to justify fraud models, or studies leakage and shortcuts mainly in other domains. This study joins the two. We use SHAP and LIME as an audit tool on a public tabular healthcare fraud dataset, test each finding with simple retraining checks, and compare the outcome with a dataset whose labels are supplied with the data.')

H1('3 Datasets')
H2('3.1 Dataset A: Healthcare Providers (rule-based labels)')
P('Dataset A is the Healthcare Providers dataset published on Kaggle {tamil}. It is derived from the Centers for Medicare & Medicaid Services (CMS) Medicare Provider Utilization and Payment Data. It contains 100,000 records with 27 columns. Each record describes one provider and one billed service (HCPCS code), with the submitted charge, the Medicare allowed and paid amounts, service and beneficiary counts, and provider descriptors such as provider type, state and entity type.')
P('The dataset has no fraud label. As in our related work on this dataset, we assign a label with a statistical rule. For each provider type, we compute the mean and standard deviation of Average Submitted Charge Amount. A record is labelled fraudulent when its charge is more than two standard deviations above the mean for its provider type. This rule labels 2,991 records (2.99%) as fraudulent. The label is therefore a fixed function of two input features: Average Submitted Charge Amount and Provider Type. This known construction lets us check whether the audit can find it.')
P('Identifier and address columns (provider name, street, city, ZIP code) and the row index were removed. Numeric columns stored as text were converted to numbers, missing numeric values were filled with the column median, and categorical columns were label-encoded. Three ratio features were added: Payment-to-Charge Ratio (PCR, Medicare payment divided by submitted charge, clipped to [0, 1]), Charge_Allowed_Ratio (submitted charge divided by allowed amount, clipped to [0, 10]) and Services_Per_Beneficiary. PCR and Charge_Allowed_Ratio are computed from the submitted charge, so they are treated as derived suspect features in the audit. A correlation filter (|r| > 0.85) then removed five features: National Provider Identifier, Number of Distinct Medicare Beneficiary/Per Day Services, Average Medicare Payment Amount, Average Medicare Standardized Amount and Payment_Per_Beneficiary. The final feature set has 19 features.')
H2('3.2 Dataset B: Healthcare Provider Fraud Detection (supplied labels)')
P('Dataset B is the Healthcare Provider Fraud Detection Analysis dataset, also published on Kaggle {gupta}. It contains Medicare claims split into inpatient claims, outpatient claims and beneficiary records, plus a provider-level label (PotentialFraud: Yes/No) supplied with the data. We used the training files only. Claims were joined with beneficiary records and aggregated to one row per provider. This gives 5,410 providers, of which 506 (9.35%) are labelled as potential fraud.')
P('Twenty provider-level features were built. They cover claim volume (number of claims, number of distinct beneficiaries, claims per beneficiary, share of inpatient claims); money (mean, total and maximum reimbursed amount, mean deductible paid); duration (mean claim length, mean admission length); coding (mean number of diagnosis and procedure codes); physicians (number of attending physicians, share of claims with an operating or other physician); and patient mix (mean age, share deceased, mean number of chronic conditions, share with renal disease, share male). The labels in dataset B are not computed from any of these features. This makes it a reference case for how the audit behaves when the label was not built from the inputs.')
H2('3.3 Data splits')
P('For both datasets, each experiment used stratified 80/20 train–test splits with five random seeds (0–4). No resampling was applied. Class imbalance was handled through class weights in the models (Section 4).')

H1('4 Explanation-Based Leakage Audit')
P('We propose a five-step audit that uses post-hoc explanations to test whether a fraud model has learned real behaviour or the rule that created the labels (Fig. 1). The audit runs after a model is trained. It needs only the training data, the model, and SHAP and LIME. We ran every step with three tree ensembles: LightGBM {ke} (300 trees, learning rate 0.05), XGBoost {chen} (300 trees, maximum depth 6) and Random Forest {breiman} (200 trees, maximum depth 14, minimum leaf size 5). Class imbalance was handled with a positive-class weight equal to the ratio of negative to positive cases (balanced class weights for Random Forest). Each experiment was repeated over five stratified 80/20 train–test splits (seeds 0–4). F1 is reported at the best threshold on a grid from 0.05 to 0.95. This threshold is chosen on the test split, so the F1 values are optimistic, but the same rule is applied to every model and feature set.')
FIG('figs/p2_fig1_audit.png', 'Fig. 1 The five-step explanation-based leakage audit. Step 2 (the suspect-only model) is the decisive test')
H2('4.1 Step 1: Attribution concentration')
P('We compute TreeSHAP values {lundberg20} for up to 2,000 test records (500 for Random Forest, because of run time). Each feature’s mean absolute SHAP value is divided by the total over all features, which gives its share of the model’s attribution. We record the top-ranked feature and the share held by a candidate set of suspect features. Suspect features are the features named in the dataset’s labelling procedure, when it is documented, plus any feature that holds a clearly dominant share. A single feature that carries most of the attribution in every model and every split is a warning sign, but it is not proof of leakage, because a real fraud signal can also dominate.')
H2('4.2 Step 2: Suspect-only model')
P('We retrain each model on the suspect features alone and compare its F1 and ROC-AUC with the full model. If a model that sees only two or three features performs as well as, or better than, the model that sees all features, the label can be predicted almost entirely from those features. The other features add nothing. We treat this as the decisive leakage test.')
H2('4.3 Step 3: Removal and proxy check')
P('We remove the suspect features and any feature computed from them, retrain, and run SHAP again. A large fall in F1 confirms that the original performance depended on the suspect features. We then look at which feature becomes the new top-ranked one. If it is closely related to a removed feature (for example, another monetary amount for the same service), it may act as a stand-in, and the leak is only partly removed.')
H2('4.4 Step 4: Global explanation stability')
P('For each model, we compare the normalised SHAP importance vectors across the five splits. For each split, we also compare them across the three models. Agreement is measured with Spearman’s ρ, Kendall’s τ and the Jaccard overlap of the top-5 features. High stability is usually read as a sign of a trustworthy explanation. In this audit we read it together with Steps 1–3: very high stability around a feature that defines the label means the models agree because they learned the same rule, not because they found a robust fraud pattern.')
H2('4.5 Step 5: Local explanation check with LIME')
P('We explain 30 test records with LIME: the 15 with the highest predicted fraud scores and 15 chosen at random. Each record is explained five times with different random seeds, using 2,000 perturbation samples and the top five features. We report how often the same feature ranks first, and the mean and minimum pairwise top-5 Jaccard overlap across the repeats. This shows whether single-case explanations point to the same suspect feature, and how much of the rest of each explanation is reliable.')
H2('4.6 Reference dataset')
P('All five steps are also run on dataset B, whose labels come with the data and are not computed from its features. On dataset B, the suspect feature for Steps 2 and 3 is the top-ranked feature from Step 1, and these two steps were run with LightGBM. Dataset B gives a reference for how concentration, stability and LIME agreement look when the label was not built from the inputs.')
H2('4.7 Use of AI tools')
P('An AI assistant (Claude, Anthropic) was used to write and run the experiment code, to draft parts of the manuscript text, and to edit its language. The authors designed the study, checked all code, results and text, and take full responsibility for the content.')

H1('5 Results')
H2('5.1 Predictive performance across feature sets')
P('Table 1 and Fig. 2 show how each model performs on dataset A with three feature sets, and on dataset B. With all features, LightGBM and XGBoost reach F1 0.95–0.96 on dataset A. Random Forest is lower (0.789), which is probably due to the depth limit used to keep SHAP run time practical. Models trained on only the two suspect features match or beat the full models: F1 is 0.947–0.960 and ROC-AUC is 0.999–1.000. For Random Forest, the suspect-only model improves F1 by 0.164. The remaining 17 features therefore add no predictive value, which is what the Step 2 leakage test looks for.')
P('Removing the suspect features cuts F1 to 0.495–0.530, a drop of 0.29–0.43 compared with the full models. ROC-AUC stays high (0.953–0.960), so the models still rank fraud cases well with the remaining features. Section 5.2 shows that this ranking relies mainly on another monetary amount. On dataset B, all three models score F1 0.639–0.650 and ROC-AUC 0.933–0.941, with larger variation across splits (SD 0.026–0.038). The smaller sample (5,410 providers) partly explains that variation. A LightGBM model trained on ReimbSum alone, the top-ranked feature on dataset B, reaches F1 0.614, below the full model, and removing ReimbSum leaves F1 unchanged (0.654).')
TAB(1)
FIG('figs/p2_fig3_f1.png', 'Fig. 2 F1 (mean ± SD over five splits) with all features, suspect features only, and suspects removed. (a) Dataset A, three models. (b) Dataset B, LightGBM, with ReimbSum as the suspect feature')
H2('5.2 Attribution concentration and proxy features')
P('Average Submitted Charge Amount ranked first in all 15 runs on dataset A (Table 2, Fig. 3). Averaged over all runs, it holds 51.1% of the attribution. Provider Type (10.1%) and Average Medicare Allowed Amount (9.7%) come next, and no other feature reaches 5%. The two features that define the label together take 44–78% of the attribution, depending on the model. The features derived from charge (PCR and Charge_Allowed_Ratio) add a further 4–16%. In the suspect-only models, charge holds 74–77% of the attribution and Provider Type the rest. This matches the labelling rule, in which charge is compared against a mean and standard deviation for each provider type.')
P('After the suspect features are removed, Average Medicare Allowed Amount ranks first in all 15 runs, with 37–50% of the attribution. Allowed amount is a second monetary value for the same service and moves closely with the submitted charge, so it acts as a partial stand-in for the removed feature. This explains why ROC-AUC stays high while F1 falls. Deleting the label-defining column reduces the leak but does not remove it.')
P('On dataset B, ReimbSum (total reimbursed amount per provider) also ranks first in every run, with 25–40% of the attribution. On average it holds 31.9%, followed by AdmitDaysMean (8.1%) and ClaimsPerBene (7.0%). So attribution is less concentrated on dataset B, but the ranges overlap: XGBoost’s top-1 share on dataset B (0.402) is higher than Random Forest’s on dataset A (0.381). Concentration alone therefore cannot separate a leaky label from a strong real signal. The suspect-only test in Step 2 is needed to decide.')
TAB(2)
FIG('figs/p2_fig2_shap.png', 'Fig. 3 Mean share of SHAP attribution for the six top features, averaged over 15 runs (3 models × 5 splits). (a) Dataset A: label-defining features in blue, a feature derived from charge in orange. (b) Dataset B: labels supplied with the data')
H2('5.3 Global explanation stability')
P('On dataset A, the explanations barely change across splits (ρ = 0.955–0.996), and the three models largely agree with each other (ρ = 0.893) (Table 3). On dataset B, stability across splits is lower for the boosting models (ρ = 0.721–0.742), and agreement between models is about half as strong (ρ = 0.445, top-5 Jaccard = 0.413). Read on its own, this would suggest the dataset A explanations are more trustworthy. Read together with Sections 5.1 and 5.2, it means the opposite: the models agree on dataset A because they all learned the same labelling rule.')
TAB(3)
H2('5.4 Local explanations with LIME')
P('On both datasets, LIME put the same feature first in every explained record, and it matched the SHAP top-ranked feature (Table 4). Beyond the first feature, LIME explanations were unstable across repeats. The mean top-5 overlap was 0.34 on dataset A and 0.47 on dataset B, so most of a record’s top-5 list changed when only LIME’s random seed changed. LIME’s agreement on the first feature confirms the global SHAP finding at the level of single records. The other features in a LIME explanation should not be read as reliable evidence.')
TAB(4)

H1('6 Discussion')
H2('6.1 Explanations as a data audit, not only a model audit')
P('SHAP and LIME are usually used to justify a model’s decisions to users. In this study, their most useful role was to question the data. On dataset A, the explanations pointed to Average Submitted Charge Amount in every run, for every model and every method. The suspect-only test then showed that two features could reproduce the labels as well as the full feature set (F1 0.947–0.960). High scores on this dataset, including those in our own earlier experiments, therefore mostly show that the model relearned how the labels were built, not that it can detect fraud. We recommend that published results on datasets whose labels are computed from the features be read with this in mind. We also recommend that authors report how the labels were created before reporting model scores.')
H2('6.2 Stable explanations are not the same as valid explanations')
P('Explanation stability is often treated as a quality measure. A stable explanation is seen as more reliable than one that changes between runs. Our results show why stability alone can mislead. On dataset A, explanations were highly stable across splits (ρ = 0.955–0.996) and across models (ρ = 0.893), but only because all models learned the same deterministic rule. On dataset B, where the labels are not a function of the features, stability was much lower (ρ = 0.445 across models). We suggest that stability be reported together with a test like Step 2. High stability around a feature that alone reproduces the label is a warning sign, not a strength.')
H2('6.3 Removing a column does not remove a leak')
P('A common fix for a suspected leak is to delete the offending feature. When we did this, F1 dropped by 0.29–0.43, but ROC-AUC stayed at 0.953–0.960. Average Medicare Allowed Amount became the top feature in every run, with 37–50% of the attribution. The allowed amount is a second monetary value for the same service and moves with the submitted charge, so it partly carries the removed signal. The remaining performance therefore cannot be read as a clean estimate of real fraud-detection ability either. A cleaner way forward is to redefine the label using information that is not among the model inputs. Examples are exclusion records such as the LEIE, or confirmed investigation outcomes.')
H2('6.4 A strong top feature is not always a leak')
P('Dataset B also has a dominant feature: ReimbSum ranked first in every run, with 25–40% of the attribution. The concentration measure in Step 1 alone cannot tell the two datasets apart, because the ranges overlap. Step 2 can. On dataset B, a LightGBM model trained on ReimbSum alone reached F1 0.614 ± 0.035, below the full model (0.650 ± 0.035). Removing ReimbSum did not lower performance at all (F1 0.654 ± 0.027). So the other features carry the same information, and no single feature defines the label. On dataset A, the opposite holds: the suspect features alone match the full model, and removing them cuts F1 by up to 0.43. The dominance of ReimbSum on dataset B more likely reflects provider size, since larger providers submit more claims and receive more money. It may also reflect how the dataset’s creators chose the flagged providers. This is a possible shortcut that deserves its own study, but it does not show the label was derived from the inputs.')
H2('6.5 What LIME adds')
P('LIME agreed with SHAP on the top feature in all 60 explained records across both datasets. This supports the global finding at the level of single cases. Beyond the first feature, LIME was unreliable: about two-thirds of a record’s top-5 list on dataset A changed when only the random seed changed. In an audit, LIME is useful as a quick per-case check of the main driver. It should not be used to argue about secondary features unless it is run several times and the results are aggregated.')
H2('6.6 Practical recommendations')
P('For researchers using public fraud datasets, we suggest the following checks before reporting model scores:')
BUL(['Document how the labels were produced.',
     'Run Steps 1 and 2 of the audit. If a small set of features matches the full model, report that result alongside the main results.',
     'After removing suspect features, run SHAP again and report any stand-in feature that takes over.',
     'Report explanation stability together with the suspect-only test, not on its own.'])

H1('7 Limitations')
BUL(['The audit was tested on two public datasets only. It still needs to be tested on data with confirmed fraud labels.',
     'The suspect features on dataset A were known from the labelling procedure. On datasets with undocumented labels, the suspect set must come from Step 1 alone, which may miss leaks spread over several features.',
     'F1 was reported at the best threshold on the test split. This inflates absolute values, although it is applied in the same way to every comparison.',
     'Random Forest depth was limited, and SHAP for Random Forest used 500 records to keep run time practical.',
     'LIME was run on 30 records with one model and one split.',
     'Only tree-based models were studied. Neural networks or linear models may spread attribution differently.',
     'The suspect-only and removal tests on dataset B used LightGBM only.'])

H1('8 Conclusion')
P('This study used SHAP and LIME to audit the data behind a fraud model, not only the model itself. On a public Medicare dataset with rule-based labels, the explanations pointed straight to the feature that defines the label. Two simple retraining checks confirmed it: the label-defining features alone matched the full model, and removing them caused a large drop in F1, with a related payment feature partly taking their place. Explanations were most stable on this leaky dataset, which shows that stability on its own is not proof of a trustworthy model. On a second dataset with supplied labels, the same audit found a dominant feature but no sign of leakage. This shows that the retraining checks, not attribution alone, separate the two cases.')
P('We recommend that studies using public fraud datasets report how their labels were produced and run at least the attribution and suspect-only checks before reporting performance. Future work will apply the audit to Medicare data with external labels from the LEIE. It will also extend the audit to neural and linear models, and test whether automatic selection of suspect features can find leaks spread across several features.')

DECL = [
 ('Funding: ', 'No funding was received for this research.'),
 ('Competing interests: ', 'The authors have no competing interests to declare that are relevant to the content of this article.'),
 ('Ethics approval: ', 'Not applicable. The study uses two publicly released, anonymised administrative datasets. No human participants were involved and no identifiable personal data were processed.'),
 ('Consent to participate: ', 'Not applicable.'),
 ('Consent for publication: ', 'Not applicable.'),
 ('Data availability: ', 'Both datasets are openly available on Kaggle: dataset A at https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data and dataset B at https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis. The labels for dataset A and all derived features can be reproduced by following Section 3.'),
 ('Code availability: ', 'The code for the audit (data preparation, experiments and figures) is publicly available at [GitHub URL to be added] as a small Python package, together with the scripts that reproduce every table and figure.'),
 ('Author contributions: ', 'Zia Ullah: conceptualization, methodology, software, formal analysis, writing (original draft). Wahab Khan: supervision, validation, writing (review and editing). Both authors read and approved the final manuscript.'),
 ('Related manuscripts: ', 'Two related manuscripts by the same authors, which use dataset A for different purposes, are under review elsewhere. They are disclosed in the cover letter.'),
]

# ---------------- citation numbering ----------------
order = []
def scan(t):
    for m in re.finditer(r'\{([a-z0-9,]+)\}', t):
        for k in m.group(1).split(','):
            assert k in REFS, k
            if k not in order: order.append(k)
for kind, pay in S:
    if kind in ('p',): scan(pay)
    if kind == 'list': [scan(x) for x in pay[0]]
assert set(order) == set(REFS), set(REFS) - set(order)
num = {k: i + 1 for i, k in enumerate(order)}
def fmt_nums(ns):
    ns = sorted(ns); out = []; i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        out.append(f'{ns[i]}–{ns[j]}' if j - i >= 2 else (', '.join(map(str, ns[i:j + 1]))))
        i = j + 1
    return '[' + ', '.join(out) + ']'
def cite(t): return re.sub(r'\{([a-z0-9,]+)\}', lambda m: fmt_nums([num[k] for k in m.group(1).split(',')]), t)

# ---------------- tables from results ----------------
def agg(key):
    d = pd.DataFrame(R[key]['runs']); return d.groupby('model')
ML = [('LGBM', 'LightGBM'), ('XGB', 'XGBoost'), ('RF', 'Random Forest')]
def table1():
    rows = [['Data / feature set', 'Model', 'F1', 'ROC-AUC', 'PR-AUC']]
    for key, lab in [('A_full', 'A, all features'), ('A_ruleonly', 'A, suspect only'), ('A_nocharge', 'A, suspects removed'), ('B', 'B, all features')]:
        g = agg(key)
        for i, (m, mn) in enumerate(ML):
            rows.append([lab if i == 0 else '', mn, f"{g.f1.mean()[m]:.3f} ± {g.f1.std()[m]:.3f}", f"{g.auc.mean()[m]:.3f}", f"{g.ap.mean()[m]:.3f}"])
    rows.append(['B, ReimbSum only', 'LightGBM', f"{B_ONLY['f1']:.3f} ± {B_ONLY['sd']:.3f}", f"{B_ONLY['auc']:.3f}", f"{B_ONLY['ap']:.3f}"])
    rows.append(['B, ReimbSum removed', 'LightGBM', f"{B_REM['f1']:.3f} ± {B_REM['sd']:.3f}", f"{B_REM['auc']:.3f}", f"{B_REM['ap']:.3f}"])
    return ('Table 1 Test performance, mean (± SD for F1) over five splits. Suspect only = Average Submitted Charge Amount and Provider Type. Suspects removed = all features except charge, PCR and Charge_Allowed_Ratio', rows)
def table2():
    rows = [['Data / feature set', 'Model', 'Top-ranked feature (runs)', 'Top-1 share', 'Rule share']]
    for key, lab in [('A_full', 'A, all features'), ('A_nocharge', 'A, suspects removed'), ('B', 'B, all features')]:
        d = pd.DataFrame(R[key]['runs'])
        for i, (m, mn) in enumerate(ML):
            dm = d[d.model == m]; top = dm.top1.value_counts()
            rs = '—' if key == 'B' else f"{dm.share_rule.mean():.3f}"
            rows.append([lab if i == 0 else '', mn, f"{top.index[0]} ({top.iloc[0]}/{len(dm)})", f"{dm.top1_share.mean():.3f}", rs])
    return ('Table 2 SHAP attribution share, mean over splits. Rule share = combined share of Average Submitted Charge Amount and Provider Type', rows)
def table3():
    rows = [['Data', 'Comparison', 'Spearman ρ', 'Kendall τ', 'Top-5 Jaccard']]
    for key, lab in [('A_full', 'A, all features'), ('A_nocharge', 'A, suspects removed'), ('B', 'B, all features')]:
        first = True
        if key != 'A_nocharge':
            for m, mn in ML:
                s = R[key]['stab_seed'][m]; rows.append([lab if first else '', f'Across splits, {mn}', f"{s['spearman']:.3f}", f"{s['kendall']:.3f}", f"{s['jaccard5']:.3f}"]); first = False
        s = R[key]['stab_model']; rows.append([lab if first else '', 'Across models', f"{s['spearman']:.3f}", f"{s['kendall']:.3f}", f"{s['jaccard5']:.3f}"])
    return ('Table 3 Agreement of normalised SHAP importance vectors. Across splits = pairs of splits within one model; across models = pairs of models within one split', rows)
def table4():
    rows = [['Data', 'Most frequent top-1 feature', 'Top-1 frequency', 'Top-5 Jaccard, mean', 'Top-5 Jaccard, min']]
    for key, lab in [('lime_A', 'A'), ('lime_B', 'B')]:
        l = R[key]; f, v = list(l['top1_freq'].items())[0]
        rows.append([lab, f, f"{round(v * l['n'])}/{l['n']}", f"{l['jaccard5_mean']:.2f}", f"{l['jaccard5_min']:.2f}"])
    return ('Table 4 LIME stability over five repeats per record (30 test records, LightGBM, split 0)', rows)
TABLES = {1: table1, 2: table2, 3: table3, 4: table4}

# ---------------- docx ----------------
doc = Document()
st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.5
for s in doc.sections: s.left_margin = s.right_margin = Cm(2.5); s.top_margin = s.bottom_margin = Cm(2.5)
sp = doc.sections[0]._sectPr; ln = OxmlElement('w:lnNumType'); ln.set(qn('w:countBy'), '1'); ln.set(qn('w:restart'), 'continuous'); sp.append(ln)

def para(text, bold=False, size=None, align=None, italic=False, after=None):
    p = doc.add_paragraph(); r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    if align is not None: p.alignment = align
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p
para(TITLE, bold=True, size=15, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
para(AUTH, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para(AFF, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, after=2)
para(CORR, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, after=12)
para('Abstract', bold=True, size=12)
for lab, t in ABS:
    p = doc.add_paragraph(); p.add_run(lab).bold = True; p.add_run(t); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p = doc.add_paragraph(); p.add_run('Keywords: ').bold = True; p.add_run(KEYWORDS)

def shade(cell, hexcol):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement('w:shd'); s.set(qn('w:val'), 'clear'); s.set(qn('w:color'), 'auto'); s.set(qn('w:fill'), hexcol); tcPr.append(s)
def add_table(cap, rows):
    c = doc.add_paragraph(); c.add_run(cap.split(' ', 2)[0] + ' ' + cap.split(' ', 2)[1] + ' ').bold = True; c.add_run(cap.split(' ', 2)[2])
    c.paragraph_format.keep_with_next = True; c.paragraph_format.line_spacing = 1.15
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = 'Table Grid'
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = t.cell(i, j); cell.text = ''; pp = cell.paragraphs[0]; r = pp.add_run(v); r.font.size = Pt(9); r.bold = (i == 0)
            pp.paragraph_format.space_after = Pt(0); pp.paragraph_format.line_spacing = 1.0
            if i == 0: shade(cell, 'E8EEF6')
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

for kind, pay in S:
    if kind == 'h1': h = doc.add_paragraph(); r = h.add_run(pay); r.bold = True; r.font.size = Pt(13); h.paragraph_format.space_before = Pt(12); h.paragraph_format.keep_with_next = True
    elif kind == 'h2': h = doc.add_paragraph(); r = h.add_run(pay); r.bold = True; r.italic = True; h.paragraph_format.space_before = Pt(6); h.paragraph_format.keep_with_next = True
    elif kind == 'p': p = doc.add_paragraph(cite(pay)); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif kind == 'list':
        items, numd = pay
        for x in items: doc.add_paragraph(cite(x), style='List Number' if numd else 'List Bullet')
    elif kind == 'table': add_table(*TABLES[pay]())
    elif kind == 'fig':
        path, cap = pay; doc.add_picture(path, width=Cm(15.5)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = doc.add_paragraph(); b, rest = cap.split(' ', 2)[:2], cap.split(' ', 2)[2]; c.add_run(' '.join(b) + ' ').bold = True; c.add_run(rest); c.paragraph_format.line_spacing = 1.15

h = doc.add_paragraph(); r = h.add_run('Statements and Declarations'); r.bold = True; r.font.size = Pt(13); h.paragraph_format.space_before = Pt(12)
for lab, t in DECL:
    p = doc.add_paragraph(); p.add_run(lab).bold = True; p.add_run(t)
h = doc.add_paragraph(); r = h.add_run('References'); r.bold = True; r.font.size = Pt(13); h.paragraph_format.space_before = Pt(12)
for k in order:
    p = doc.add_paragraph(f'{num[k]}. {REFS[k]}'); p.paragraph_format.line_spacing = 1.15; p.paragraph_format.left_indent = Cm(0.7); p.paragraph_format.first_line_indent = Cm(-0.7)
    for r in p.runs: r.font.size = Pt(10)
out = 'Paper2_LeakageAudit_ZiaUllah_DRAFT.docx'; doc.save(out)
print('saved', out, 'refs', len(order))
for i, k in enumerate(order, 1): print(i, k)
