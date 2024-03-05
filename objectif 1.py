#TD Analyse de sentiments TRAITEMENT DU LANGAGE NATUREL

#Objectif 1

#Import des librairies nécéssaires
import xml.etree.ElementTree as ET
import spacy
import matplotlib.pyplot as plt
import spacy
from collections import defaultdict
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import sentiwordnet as swn


# Chemins des datasets
datasets_base_path = "datasets/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/"
laptop_train_path = datasets_base_path + "Laptop_Train.xml"
laptop_test_gold_path = datasets_base_path + "Laptop_Test_Gold.xml"
laptop_test_nolabels_path = datasets_base_path + "Laptop_Test_NoLabels.xml"

restaurant_train_path = datasets_base_path + "Restaurants_Train.xml"
restaurant_test_gold_path = datasets_base_path + "Restaurants_Test_Gold.xml"
restaurant_test_nolabels_path = datasets_base_path + "Restaurants_Test_NoLabels.xml"

# Chemin de l'EmoLex
emolex_path = "NRC-Emotion-Lexicon\\NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# Chargement du modèle spaCy
nlp = spacy.load("en_core_web_sm")



#Objectif 1 : calculer la polarité des mots dans les deux jeux de données à l’aide d’un lexicon de sentiment.

#Pré-traitement des phrases
def preprocess_text(text):
    # Utiliser spaCy pour traiter le texte
    doc = nlp(text)
    
    # Tokenisation et PoS tagging
    tokens = [(token.text, token.pos_) for token in doc]

    # Reconnaissance d'entités nommées (NER)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Gestion de la négation: Identifier les tokens qui sont des négations
    negations = [token.text for token in doc if token.dep_ == 'neg']
    
    # Retourner un dictionnaire avec les informations extraites
    return {
        'tokens': tokens,
        'entities': entities,
        'negations': negations
    }

# Exemple d'utilisation
text_example = "The food was not good and the service was terrible."
preprocessed = preprocess_text(text_example)

#print(preprocessed)

#Identification des polaritées 
# Charger EmoLex dans un dictionnaire
emotion_lexicon = defaultdict(lambda: {'positive': 0, 'negative': 0})
with open(emolex_path, 'r', encoding='utf-8') as file:
    for line in file:
        word, emotion, association = line.strip().split('\t')
        if emotion in ['positive', 'negative']:
            emotion_lexicon[word][emotion] = int(association)

# Fonction pour analyser la polarité des mots d'une phrase
def analyze_word_polarity(text):
    doc = nlp(text)
    word_polarity = []

    for token in doc:
        # Ignorer les ponctuations et les stop words
        if token.is_punct or token.is_stop:
            continue
        # Chercher la polarité du mot dans EmoLex
        if token.lemma_ in emotion_lexicon:
            word_polarity.append((token.text, emotion_lexicon[token.lemma_]))

    return word_polarity

# Fonction pour analyser la polarité des mots dans un fichier XML
def analyze_file_polarity(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    polarity_count = {'positive': 0, 'negative': 0}

    for sentence in root.findall('.//sentence'):
        text = sentence.find('text').text
        word_polarities = analyze_word_polarity(text)
        for _, polarities in word_polarities:
            polarity_count['positive'] += polarities['positive']
            polarity_count['negative'] += polarities['negative']
    
    return polarity_count

# Exemple d'utilisation
text = "The food was great but the service was bad."
polarity = analyze_word_polarity(text)

#print(polarity)


# Analyse de polarité pour chaque fichier et stockage des résultats
file_paths = [laptop_train_path, laptop_test_gold_path, laptop_test_nolabels_path, restaurant_train_path, restaurant_test_gold_path, restaurant_test_nolabels_path]
all_polarity_counts = {}

for file_path in file_paths:
    all_polarity_counts[file_path] = analyze_file_polarity(file_path)

# Nombre de graphiques à afficher
num_files = len(file_paths)
# Organisation des subplots
num_rows = 2 
num_cols = 3 

# Création d'une figure pour tous les graphiques
plt.figure(figsize=(20, 10))  # Ajuster la taille globale de la figure

for i, (file_path, polarity_count) in enumerate(all_polarity_counts.items(), start=1):
    title = file_path.replace(datasets_base_path, "")
    labels = list(polarity_count.keys())
    values = list(polarity_count.values())

    # Création d'un subplot pour chaque graphique
    plt.subplot(num_rows, num_cols, i)
    plt.bar(labels, values, color=['green', 'red'])
    plt.title(f'Polarity Distribution in {title}')
    plt.xlabel('Polarity')
    plt.ylabel('Count')

# Ajustement automatique de l'espace entre les graphiques pour une meilleure lisibilité
plt.tight_layout()

# Affichage de la figure contenant tous les subplots
plt.show()























import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import sentiwordnet as swn

nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('punkt')

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []

    for sentence in root.findall('.//sentence'):
        text = sentence.find('text').text
        aspect_terms_element = sentence.find('aspectTerms')
        aspect_categories_element = sentence.find('aspectCategories')
        
        if aspect_terms_element is not None:
            aspect_terms = [
                (at.get('term'), at.get('polarity'), int(at.get('from')), int(at.get('to')))
                for at in aspect_terms_element.findall('aspectTerm')
            ]
        else:
            aspect_terms = []
        
        if aspect_categories_element is not None:
            aspect_categories = [
                (ac.get('category'), ac.get('polarity'))
                for ac in aspect_categories_element.findall('aspectCategory')
            ]
        else:
            aspect_categories = []
        
        data.append((text, aspect_terms, aspect_categories))
    
    return data

def adjust_sentiment_score_based_on_pos(word, pos_tag, sentiment_score):
    """
    Adjusts the sentiment score of a word based on its part-of-speech tag.
    """
    if pos_tag.startswith('JJ'):  # Adjective
        return sentiment_score * 1.5
    elif pos_tag.startswith('RB'):  # Adverb
        return sentiment_score * 1.2
    elif pos_tag.startswith('NN') or pos_tag.startswith('VB'):  # Noun or Verb
        return sentiment_score
    else:
        return sentiment_score * 0.8  # Other parts of speech have less impact

def calculate_sentiment(aspect, sentence, window_size=3):
    words = nltk.word_tokenize(sentence)
    aspect_tokens = nltk.word_tokenize(aspect)
    pos_tags = nltk.pos_tag(words)

    aspect_index = None
    for i in range(len(words) - len(aspect_tokens) + 1):
        if words[i:i+len(aspect_tokens)] == aspect_tokens:
            aspect_index = i
            break

    if aspect_index is None:
        return 0

    start_index = max(0, aspect_index - window_size)
    end_index = min(len(words), aspect_index + len(aspect_tokens) + window_size)
    sentiment_score = 0

    for i in range(start_index, end_index):
        word = words[i]
        pos_tag = pos_tags[i][1]
        synsets = list(swn.senti_synsets(word))
        if synsets:
            base_score = synsets[0].pos_score() - synsets[0].neg_score()
            adjusted_score = adjust_sentiment_score_based_on_pos(word, pos_tag, base_score)
            sentiment_score += adjusted_score

    return sentiment_score

def determine_polarity(sentiment_score):
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

def evaluate_predictions(predictions, gold_standard):
    # Assume predictions and gold_standard are dictionaries with aspect terms as keys
    # and sentiment polarities ('positive', 'negative', 'neutral') as values.
    true_positives = {k: 0 for k in predictions.keys()}
    false_positives = {k: 0 for k in predictions.keys()}
    false_negatives = {k: 0 for k in predictions.keys()}
    
    for aspect, polarity in predictions.items():
        if aspect in gold_standard and polarity == gold_standard[aspect]:
            true_positives[aspect] += 1
        elif aspect not in gold_standard or polarity != gold_standard[aspect]:
            false_positives[aspect] += 1
        if aspect in gold_standard and polarity != gold_standard[aspect]:
            false_negatives[aspect] += 1
    
    accuracy = sum(true_positives.values()) / len(predictions)
    recall = sum(true_positives.values()) / (sum(true_positives.values()) + sum(false_negatives.values()))
    F_measure = 2 * (accuracy * recall) / (accuracy + recall)
    
    return accuracy, recall, F_measure

def process_files(train_files, test_files_no_labels, test_files_gold):
    # Predictions is a dict with key: (aspect, sentence_id) to handle multiple aspects and sentences
    predictions = {}
    sentence_id = 0  # Initialize sentence_id if you need to uniquely identify sentences

    # Process test files without labels to predict sentiment
    for file_path in test_files_no_labels:
        data = parse_xml(file_path)
        for sentence_data in data:  # Iterate directly through the list
            sentence, aspects, _ = sentence_data  # Assuming you don't need aspect categories here
            for aspect_term, polarity, _, _ in aspects:  # Unpack aspect term details
                sentiment_score = calculate_sentiment(aspect_term, sentence)
                predictions[(aspect_term, sentence_id)] = determine_polarity(sentiment_score)
            sentence_id += 1  # Increment sentence_id for unique identification

    # Load gold standard for evaluation
    gold_standard = {}
    sentence_id = 0  # Reset or reuse sentence_id for consistency
    for file_path in test_files_gold:
        gold_data = parse_xml(file_path)
        for sentence_data in gold_data:
            sentence, aspects, _ = sentence_data
            for aspect_term, polarity, _, _ in aspects:
                gold_standard[(aspect_term, sentence_id)] = polarity  # Assuming you're using the gold standard polarity directly
            sentence_id += 1

    # Evaluate predictions against the gold standard
    accuracy, recall, F_measure = evaluate_predictions(predictions, gold_standard)

    # Print the evaluation results
    print(f'Accuracy: {accuracy:.4f}\nRecall: {recall:.4f}\nF-measure: {F_measure:.4f}')


# Example usage:
train_files = ["SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train.xml", "SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train.xml"]
test_files_no_labels = ["SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Test_NoLabels.xml", "SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Test_NoLabels.xml"]
test_files_gold = ["SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Test_Gold.xml", "SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Test_Gold.xml"]
process_files(train_files, test_files_no_labels, test_files_gold)
