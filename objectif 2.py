import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import sentiwordnet as swn

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

import xml.etree.ElementTree as ET

def extract_from_xml(xml_path):

    def parse_aspect_terms(aspect_terms_element):

        return [
            (term.get('term'), term.get('polarity'), int(term.get('from')), int(term.get('to')))
            for term in aspect_terms_element.findall('aspectTerm')
        ] if aspect_terms_element is not None else []

    def parse_aspect_categories(aspect_categories_element):

        return [
            (category.get('category'), category.get('polarity'))
            for category in aspect_categories_element.findall('aspectCategory')
        ] if aspect_categories_element is not None else []

    # Parse XML
    document_tree = ET.parse(xml_path)
    document_root = document_tree.getroot()
    extracted_data = []

    for sentence in document_root.findall('.//sentence'):
        sentence_text = sentence.find('text').text
        aspect_terms = parse_aspect_terms(sentence.find('aspectTerms'))
        aspect_categories = parse_aspect_categories(sentence.find('aspectCategories'))

        extracted_data.append((sentence_text, aspect_terms, aspect_categories))

    return extracted_data


def modify_sentiment_score_by_pos(word, pos_tag, original_sentiment_score):

    pos_score_modifiers = {
        'JJ': 1.5,  # Adjectif
        'RB': 1.2,  # Adverbe
        'NN': 1.0,  # Nom
        'VB': 1.0   # Verbe
    }

    default_modifier = 0.8

    for pos_prefix, modifier in pos_score_modifiers.items():
        if pos_tag.startswith(pos_prefix):
            return original_sentiment_score * modifier
    
    # Si l'étiquette grammaticale ne correspond à aucune catégorie spécifiée, appliquer le modificateur par défaut
    return original_sentiment_score * default_modifier



def get_adjusted_sentiment_score(word, part_of_speech):

    synsets = list(swn.senti_synsets(word))
    if not synsets:
        return 0

    base_score = synsets[0].pos_score() - synsets[0].neg_score()
    return modify_sentiment_score_by_pos(word, part_of_speech, base_score)

def find_aspect_position_in_sentence(aspect_tokens, sentence_tokens):

    for index in range(len(sentence_tokens) - len(aspect_tokens) + 1):
        if sentence_tokens[index:index+len(aspect_tokens)] == aspect_tokens:
            return index
    return None

def compute_aspect_sentiment(aspect, sentence, context_window=3):

    sentence_tokens = nltk.word_tokenize(sentence)
    aspect_tokens = nltk.word_tokenize(aspect)
    pos_tags = nltk.pos_tag(sentence_tokens)

    aspect_start_index = find_aspect_position_in_sentence(aspect_tokens, sentence_tokens)
    if aspect_start_index is None:
        return 0

    sentiment_score = 0
    context_start = max(0, aspect_start_index - context_window)
    context_end = min(len(sentence_tokens), aspect_start_index + len(aspect_tokens) + context_window)

    for i in range(context_start, context_end):
        word, pos_tag = sentence_tokens[i], pos_tags[i][1]
        sentiment_score += get_adjusted_sentiment_score(word, pos_tag)

    return sentiment_score

def get_sentiment_polarity(score, thresholds={'positive': 0, 'negative': 0}):
    if score > thresholds['positive']:
        return 'positive'
    elif score < thresholds['negative']:
        return 'negative'
    else:
        return 'neutral'


def calculate_evaluation_metrics(predicted_sentiments, actual_sentiments):

    metrics = {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}

    # Calcule des vrais positifs, faux positifs et faux négatifs
    for aspect, predicted_polarity in predicted_sentiments.items():
        actual_polarity = actual_sentiments.get(aspect)
        if predicted_polarity == actual_polarity:
            metrics['true_positives'] += 1
        else:
            metrics['false_positives'] += 1
            if actual_polarity is not None:
                metrics['false_negatives'] += 1

    # Calcul des métriques d'évaluations
    accuracy = metrics['true_positives'] / len(predicted_sentiments)
    precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
    recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
    F1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, recall, F1_score


def process_files(training_data_paths, unlabeled_test_data_paths, labeled_test_data_paths):

    # Préparation la prédiction de sentiment
    predictions, sentence_counter = {}, 0

    # Prédiction le sentiment pour les données de test non étiquetées
    for test_path in unlabeled_test_data_paths:
        test_data = extract_from_xml(test_path)
        for sentence, aspects, _ in test_data:
            for aspect_details in aspects:
                aspect_term = aspect_details[0]  # Extraction des aspects
                sentiment_score = compute_aspect_sentiment(aspect_term, sentence)
                prediction_key = (aspect_term, sentence_counter)
                predictions[prediction_key] = get_sentiment_polarity(sentiment_score)
            sentence_counter += 1

    # Charger les données de référence pour l'évaluation
    gold_standard, sentence_counter = {}, 0
    for gold_path in labeled_test_data_paths:
        gold_data = extract_from_xml(gold_path)
        for sentence, aspects, _ in gold_data:
            for aspect_details in aspects:
                aspect_term, true_polarity = aspect_details[0], aspect_details[1]
                gold_standard[(aspect_term, sentence_counter)] = true_polarity
            sentence_counter += 1

    # Évaluer les prédictions
    accuracy, recall, F1_score = calculate_evaluation_metrics(predictions, gold_standard)

    # Afficher les résultats de l'évaluation
    print(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1-score: {F1_score:.4f}')



# Chemins des datasets
datasets_base_path = "datasets/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/"
laptop_train_path = datasets_base_path + "Laptop_Train.xml"
laptop_test_gold_path = datasets_base_path + "Laptop_Test_Gold.xml"
laptop_test_nolabels_path = datasets_base_path + "Laptop_Test_NoLabels.xml"

restaurant_train_path = datasets_base_path + "Restaurants_Train.xml"
restaurant_test_gold_path = datasets_base_path + "Restaurants_Test_Gold.xml"
restaurant_test_nolabels_path = datasets_base_path + "Restaurants_Test_NoLabels.xml"

# Example usage:
train_files = [restaurant_train_path, laptop_train_path]
test_files_no_labels = [restaurant_test_nolabels_path, laptop_test_nolabels_path]
test_files_gold = [restaurant_test_gold_path, laptop_test_gold_path]
process_files(train_files, test_files_no_labels, test_files_gold)
