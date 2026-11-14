from collections import Counter
from pathlib import Path
from math import log, exp, pi
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ROWS = [line.strip().split() for line in open(ROOT / 'all_sentiment_shuffled.txt', encoding='utf-8', errors='replace') if line.strip()]
DOCS = [r[3:] for r in ROWS]
SENTIMENT = [r[1] for r in ROWS]
TOPICS = [r[0] for r in ROWS]
CUT = int(0.8 * len(DOCS))
XTRAIN, XTEST = DOCS[:CUT], DOCS[CUT:]
YTRAIN, YTEST = SENTIMENT[:CUT], SENTIMENT[CUT:]


def train_nb(documents, labels, alpha=1.0):
    """Fit multinomial naive Bayes: class priors and add-alpha token likelihoods."""
    n_class = Counter(labels)
    counts = {label: Counter() for label in n_class}
    for doc, label in zip(documents, labels):
        counts[label].update(doc)
    vocab = set().union(*(set(c) for c in counts.values()))
    totals = {label: sum(c.values()) for label, c in counts.items()}
    priors = {label: log(n / len(labels)) for label, n in n_class.items()}
    denominators = {label: totals[label] + alpha * len(vocab) for label in counts}
    return {'counts': counts, 'vocab': vocab, 'denom': denominators,
            'prior': priors, 'alpha': alpha}


def score_doc_label(document, label, model):
    """Log joint probability; never-seen vocabulary words contribute nothing."""
    score = model['prior'][label]
    counts = model['counts'][label]
    denom = model['denom'][label]
    alpha = model['alpha']
    for word, n in Counter(document).items():
        if word in model['vocab']:
            score += n * log((counts[word] + alpha) / denom)
    return score


def classify_nb(document, model):
    return max(model['prior'], key=lambda label: score_doc_label(document, label, model))


def classify_documents(documents, model):
    return [classify_nb(doc, model) for doc in documents]


def accuracy(true, pred):
    return sum(a == b for a, b in zip(true, pred)) / len(true)


def print_top_nb(k, model):
    """Stable interpretable log-odds; exclude ultra-rare tokens."""
    positives = model['counts']['pos']
    negatives = model['counts']['neg']
    scores = []
    for word in model['vocab']:
        if positives[word] + negatives[word] >= 25 and word.isalpha() and len(word) > 2:
            odds = log((positives[word] + 10) / (model['denom']['pos'] + 9 * len(model['vocab']))) - log((negatives[word] + 10) / (model['denom']['neg'] + 9 * len(model['vocab'])))
            scores.append((odds, word))
    scores.sort()
    print('Negative words:', ', '.join(w for _, w in scores[:k]))
    print('Positive words:', ', '.join(w for _, w in scores[-k:][::-1]))
    return scores


def enrich(doc, negation=False, bigrams=False):
    """Add optional marked words after negators and/or adjacent word bigrams."""
    words = []
    active = False
    for w in doc:
        if w in {'.', '!', '?', ';', ':'}:
            active = False
        if w in {'not', "n't", 'never', 'no', 'dont', "don't", 'cannot', "can't", 'without'}:
            active = True
        words.append(w + '_NEG' if active and w.isalpha() and w not in {'not', 'never', 'no'} else w)
    if not negation:
        words = list(doc)
    if bigrams:
        return words + [words[i] + '__' + words[i + 1] for i in range(len(words) - 1)]
    return words


def train_gaussian_nb(X, labels):
    """Gaussian NB from scratch, with maximum-likelihood class variance."""
    X = np.asarray(X, dtype=float)
    classes = sorted(set(labels))
    data = {}
    for c in classes:
        subset = X[np.asarray(labels) == c]
        data[c] = (subset.mean(axis=0), np.maximum(subset.var(axis=0), 1e-9), log(len(subset) / len(X)))
    return data


def score_gaussian(row, label, model):
    mean, var, prior = model[label]
    return float(prior - 0.5 * np.sum(np.log(2 * pi * var) + (row - mean) ** 2 / var))


def predict_gaussian(row, model):
    return max(model, key=lambda c: score_gaussian(np.asarray(row), c, model))


print('REQUIRED: SENTIMENT CLASSIFIER')
base = train_nb(XTRAIN, YTRAIN)
preds = classify_documents(XTEST, base)
base_acc = accuracy(YTEST, preds)
print(f'Reviews: {len(DOCS)}, train: {len(XTRAIN)}, test: {len(XTEST)}, vocabulary: {len(base["vocab"])}')
print(f'Accuracy: {base_acc:.4%} ({sum(a == b for a,b in zip(YTEST,preds))}/{len(YTEST)})')
for w in ['great', 'bad']:
    print(f'{w} P(pos,word)={exp(score_doc_label([w],"pos",base)):.6f}, P(neg,word)={exp(score_doc_label([w],"neg",base)):.6f}')
print('Unseen example:', classify_nb(['a', 'top-quality', 'performance'], base))
errors = []
for i,(doc,truth,guess) in enumerate(zip(XTEST,YTEST,preds)):
    if truth != guess:
        margin = score_doc_label(doc, guess, base) - score_doc_label(doc, truth, base)
        errors.append((margin, i, truth, guess, ' '.join(doc[:85])))
print('Wrong predictions:', len(errors))
for margin, i, truth, guess, excerpt in sorted(errors, reverse=True)[:5]:
    print(f'  index {i}, {truth} -> {guess}, log margin {margin:.1f}: {excerpt[:270]}')

print('\n1: SIX TOPIC CATEGORIES')
topic_model = train_nb(XTRAIN, TOPICS[:CUT])
topic_pred = classify_documents(XTEST, topic_model)
topic_acc = accuracy(TOPICS[CUT:], topic_pred)
print('Six-category accuracy:', f'{topic_acc:.4%}')
for topic in sorted(set(TOPICS)):
    indices = [i for i, t in enumerate(TOPICS[CUT:]) if t == topic]
    print(f'  {topic}: {sum(topic_pred[i] == topic for i in indices)}/{len(indices)} = {sum(topic_pred[i] == topic for i in indices)/len(indices):.2%}')

print('\n2: LEARNING CURVE')
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
curve = []
for f in fractions:
    n = int(f * CUT)
    score = accuracy(YTEST, classify_documents(XTEST, train_nb(XTRAIN[:n], YTRAIN[:n])))
    curve.append(score)
    print(f'  {f:.0%} train = {score:.2%}')
plt.figure(figsize=(7.2, 4.4))
plt.plot([int(CUT*f) for f in fractions], np.array(curve)*100, marker='o')
plt.xlabel('Number of training reviews')
plt.ylabel('Held-out sentiment accuracy (%)')
plt.title('Naive Bayes learning curve')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(ROOT/'../media/learning_curve.png', dpi=170)
plt.close()

print('\n3: INFORMATIVE WORDS AND VISUALIZATION')
ranked = print_top_nb(15, base)
chosen = ranked[:12] + ranked[-12:]
fig, ax = plt.subplots(figsize=(8.7, 6.4))
ax.barh([w for _,w in chosen], [s for s,_ in chosen], color=['#c65b62' if s < 0 else '#238e85' for s,_ in chosen])
ax.set_xlabel('Smoothed log likelihood ratio: positive versus negative')
ax.set_title('Indicative sentiment tokens (minimum 25 uses)')
ax.axvline(0, color='black', lw=0.8)
fig.tight_layout()
fig.savefig(ROOT/'../media/informative_words.png', dpi=170)
plt.close(fig)

print('\n4: GAUSSIAN NAIVE BAYES ON IRIS')
# sklearn is only a dataset loader here; the Gaussian classifier above is my own implementation.
from sklearn.datasets import load_iris
iris = load_iris()
rng = np.random.default_rng(42)
indices = rng.permutation(len(iris.target))
itrain, itest = indices[:110], indices[110:]
gmodel = train_gaussian_nb(iris.data[itrain], iris.target[itrain].tolist())
gpred = [predict_gaussian(row,gmodel) for row in iris.data[itest]]
gaussian_acc = accuracy(iris.target[itest], gpred)
print(f'Iris Gaussian NB: {gaussian_acc:.2%} ({sum(a==b for a,b in zip(iris.target[itest],gpred))}/40)')

print('\n5: SCIKIT-LEARN ALTERNATIVE CLASSIFIER')
# Only this optional task uses a library classifier.
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2
train_text = [' '.join(doc) for doc in XTRAIN]
test_text = [' '.join(doc) for doc in XTEST]
svc = make_pipeline(TfidfVectorizer(token_pattern=r'(?u)\b\w+\b'), LinearSVC(random_state=42))
svc.fit(train_text, YTRAIN)
svc_acc = accuracy(YTEST, svc.predict(test_text))
print(f'TF-IDF + LinearSVC accuracy: {svc_acc:.2%}')
mlp = make_pipeline(TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=12000),
                    SelectKBest(chi2, k=400),
                    MLPClassifier(hidden_layer_sizes=(32,), max_iter=35, random_state=42, early_stopping=True))
mlp.fit(train_text, [int(y == "pos") for y in YTRAIN])
mlp_acc = accuracy([int(y == "pos") for y in YTEST], mlp.predict(test_text))
print(f'TF-IDF + 400 chi-square features + MLP accuracy: {mlp_acc:.2%}')

print('\n6: IMPROVE FEATURES (NEGATION / BIGRAMS)')
feature_results = {}
feature_predictions = {}
for name, neg, bi in [('unigrams',False,False), ('negation',True,False),('bigrams',False,True),('both',True,True)]:
    tx = [enrich(doc, neg, bi) for doc in XTRAIN] if name != 'unigrams' else XTRAIN
    te = [enrich(doc, neg, bi) for doc in XTEST] if name != 'unigrams' else XTEST
    feature_pred = classify_documents(te, train_nb(tx,YTRAIN))
    feature_predictions[name] = feature_pred
    acc = accuracy(YTEST, feature_pred)
    feature_results[name] = acc
    print(f'  {name}: {acc:.2%}')

# Paired comparison on exactly the same held-out reviews.
from scipy.stats import binomtest
base_only = sum(a == t and b != t for a,b,t in zip(preds, feature_predictions['both'],YTEST))
improved_only = sum(a != t and b == t for a,b,t in zip(preds, feature_predictions['both'],YTEST))
p_value = binomtest(min(base_only, improved_only), base_only + improved_only, 0.5).pvalue
print(f'Paired exact McNemar test: baseline-only correct={base_only}, enriched-only correct={improved_only}, p={p_value:.5g}')

print('\n7: CROSS-DOMAIN SENSITIVITY')
from itertools import product
domain_results = {}
for source, target in product(['books','camera'], repeat=2):
    source_i = [i for i,t in enumerate(TOPICS[:CUT]) if t == source]
    target_i = [i for i,t in enumerate(TOPICS[CUT:]) if t == target]
    dm = train_nb([XTRAIN[i] for i in source_i], [YTRAIN[i] for i in source_i])
    da = accuracy([YTEST[i] for i in target_i],classify_documents([XTEST[i] for i in target_i],dm))
    domain_results[f'{source}_to_{target}'] = da
    print(f'  {source} -> {target}: {da:.2%} (train {len(source_i)}, test {len(target_i)})')

results = {'total': len(DOCS), 'train': CUT, 'test': len(XTEST), 'vocabulary': len(base['vocab']),
           'sentiment_accuracy': base_acc, 'topic_accuracy': topic_acc, 'learning_curve': dict(zip([str(int(CUT*f)) for f in fractions], curve)),
           'gaussian_iris_accuracy': gaussian_acc, 'linear_svc_accuracy': svc_acc, 'mlp_accuracy': mlp_acc,
           'feature_comparison': feature_results, 'domain_comparison': domain_results,
           'paired_test': {'baseline_only_correct': base_only, 'enhanced_only_correct': improved_only, 'p_value': p_value},
           'error_examples': [{'margin':m, 'index':i, 'actual':a, 'predicted':p, 'excerpt':d} for m,i,a,p,d in sorted(errors,reverse=True)[:5]]}
(ROOT/'results_all.json').write_text(json.dumps(results, indent=2))
print('\nSaved results_all.json, learning_curve.png, informative_words.png')
