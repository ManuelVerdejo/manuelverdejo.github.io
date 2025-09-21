# Sistema Avanzado de Análisis de Sentimientos
# Implementa conceptos de Minería de Opiniones Basadas en Aspectos y Técnicas Comparativas
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import math

# Descargar recursos necesarios de NLTK
def download_nltk_resources():
    """Descarga recursos de NLTK de forma robusta"""
    resources = {
        'stopwords': 'corpora/stopwords',
        'movie_reviews': 'corpora/movie_reviews', 
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
        'wordnet': 'corpora/wordnet'
    }
    
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                print(f"Descargando {resource}...")
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Error al descargar {resource}: {e}")
                continue

# Ejecutar descarga
download_nltk_resources()

class AspectBasedSentimentAnalyzer:
    """
    Implementa minería de opiniones basadas en aspectos según el modelo de 5-tuplas:
    (entidad, aspecto, orientación, titular, tiempo)
    """
    
    def __init__(self):
        # Palabras de referencia para orientación semántica (PMI) - EXPANDIDAS
        self.positive_seeds = [
            # Palabras directas positivas
            'excellent', 'amazing', 'great', 'wonderful', 'fantastic', 'perfect', 
            'outstanding', 'brilliant', 'superb', 'magnificent', 'exceptional',
            # Términos comparativos positivos - NUEVA MEJORA
            'better', 'superior', 'improved', 'enhanced', 'upgraded', 'advanced',
            'finest', 'top', 'premium', 'higher', 'greater', 'stronger'
        ]
        self.negative_seeds = [
            # Palabras directas negativas
            'poor', 'terrible', 'awful', 'horrible', 'bad', 'worst',
            'dreadful', 'appalling', 'disappointing', 'mediocre', 'subpar',
            # Términos comparativos negativos - NUEVA MEJORA
            'worse', 'inferior', 'weaker', 'lower', 'lesser', 'reduced',
            'downgraded', 'declined', 'deteriorated'
        ]
        
        # Palabras clave para oraciones comparativas (según el documento)
        self.comparative_keywords = [
            'better', 'worse', 'best', 'worst', 'superior', 'inferior',
            'greater', 'less', 'more', 'most', 'least', 'same', 'similar',
            'different', 'ahead', 'exceed', 'wins', 'number one'
        ]
        
        # Palabras de negación para mejorar la clasificación
        self.negation_words = ['not', 'never', 'no', 'none', 'neither', 'nothing', 
                              'nobody', 'nowhere', 'hardly', 'scarcely', 'barely']
        
        self.stop_words = set(stopwords.words('english'))
        
        # NUEVA MEJORA: Pesos para ajustar clasificación de comparaciones
        self.comparative_sentiment_adjustments = {
            'better': 2.0,     # Fuertemente positivo
            'best': 2.5,       # Muy positivo
            'superior': 2.0,   # Fuertemente positivo
            'improved': 1.5,   # Positivo
            'enhanced': 1.5,   # Positivo
            'worse': -2.0,     # Fuertemente negativo
            'worst': -2.5,     # Muy negativo
            'inferior': -2.0,  # Fuertemente negativo
            'declined': -1.5,  # Negativo
            'deteriorated': -1.5  # Negativo
        }
        
    def preprocess_text(self, text):
        """Preprocesamiento avanzado del texto"""
        document = str(text)
        # Mantener signos de puntuación relevantes para el análisis
        document = re.sub(r'[^a-zA-Z\s\.\!\?]', ' ', document)
        document = re.sub(r'\s+', ' ', document)
        return document.lower().strip()
    
    def extract_aspects_and_opinions(self, text):
        """
        Extrae aspectos y opiniones usando POS tagging
        Implementa el Paso 1 del algoritmo no supervisado del documento
        """
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            print(f"Error en segmentación de oraciones: {e}")
            # Fallback a división por puntos
            sentences = text.split('.')
        
        aspects_opinions = []
        
        for sentence in sentences:
            try:
                tokens = word_tokenize(sentence.lower())
                pos_tags = pos_tag(tokens)
            except Exception as e:
                print(f"Error en tokenización/POS tagging: {e}")
                # Fallback simple sin POS tagging
                tokens = sentence.lower().split()
                pos_tags = [(token, 'NN') for token in tokens]  # Asumir sustantivos
            
            # Extraer adjetivo + sustantivo y adverbio + adjetivo
            for i in range(len(pos_tags) - 1):
                current_word, current_pos = pos_tags[i]
                next_word, next_pos = pos_tags[i + 1]
                
                # Adjetivo seguido de sustantivo
                if current_pos.startswith('JJ') and next_pos.startswith('NN'):
                    aspects_opinions.append((f"{current_word} {next_word}", 'aspect_opinion'))
                
                # Adverbio seguido de adjetivo
                elif current_pos.startswith('RB') and next_pos.startswith('JJ'):
                    aspects_opinions.append((f"{current_word} {next_word}", 'opinion_modifier'))
        
        return aspects_opinions
    
    def calculate_semantic_orientation(self, phrase, corpus_stats=None):
        """
        Calcula la orientación semántica usando PMI MEJORADO
        Implementa las ecuaciones (1), (2) y (3) del documento con mejoras para comparaciones
        """
        phrase_lower = phrase.lower()
        
        if corpus_stats is None:
            # MEJORA: Detección mejorada de patrones positivos y negativos
            positive_score = 0
            negative_score = 0
            
            # Contar palabras de semilla directas
            for word in self.positive_seeds:
                if word in phrase_lower:
                    positive_score += 2  # Peso doble para coincidencias exactas
            
            for word in self.negative_seeds:
                if word in phrase_lower:
                    negative_score += 2
            
            # NUEVA MEJORA: Patrones comparativos contextuales
            comparative_positive_patterns = [
                ('better than', 3), ('superior to', 3), ('improved from', 2),
                ('better', 1.5), ('superior', 1.5), ('improved', 1.5)
            ]
            
            comparative_negative_patterns = [
                ('worse than', 3), ('inferior to', 3), ('declined from', 2),
                ('worse', 1.5), ('inferior', 1.5), ('deteriorated', 1.5)
            ]
            
            # Buscar patrones comparativos positivos
            for pattern, weight in comparative_positive_patterns:
                if pattern in phrase_lower:
                    positive_score += weight
            
            # Buscar patrones comparativos negativos  
            for pattern, weight in comparative_negative_patterns:
                if pattern in phrase_lower:
                    negative_score += weight
            
            # MEJORA: Considerar adjetivos intensificadores
            intensifiers = ['very', 'extremely', 'incredibly', 'remarkably', 'truly', 'really']
            intensifier_boost = 1.0
            for intensifier in intensifiers:
                if intensifier in phrase_lower:
                    intensifier_boost = 1.5
                    break
            
            positive_score *= intensifier_boost
            negative_score *= intensifier_boost
            
            # Evitar división por cero añadiendo 0.01 como menciona el documento
            positive_hits = max(positive_score * 100 + 0.01, 0.01)
            negative_hits = max(negative_score * 100 + 0.01, 0.01)
            
            # Ecuación (3) del documento: SO(phrase) = log2(hits(phrase NEAR "excellent") / hits(phrase NEAR "poor"))
            semantic_orientation = math.log2(positive_hits / negative_hits)
            return semantic_orientation
        
        return 0
    
    def detect_comparative_sentences(self, text):
        """
        Detecta oraciones comparativas usando las palabras clave del documento
        Implementa la detección de los 4 tipos de comparación
        """
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            print(f"Error en segmentación: {e}")
            # Fallback a división por puntos
            sentences = text.split('.')
            
        comparative_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in self.comparative_keywords:
                if keyword in sentence_lower:
                    # Clasificar tipo de comparación
                    if any(word in sentence_lower for word in ['better', 'worse', 'superior', 'inferior']):
                        comp_type = 'Desigual'
                    elif any(word in sentence_lower for word in ['same', 'similar', 'equal']):
                        comp_type = 'Equitativa'
                    elif any(word in sentence_lower for word in ['best', 'worst', 'most', 'least']):
                        comp_type = 'Superlativa'
                    else:
                        comp_type = 'No gradual'
                    
                    comparative_sentences.append({
                        'sentence': sentence,
                        'type': comp_type,
                        'keyword': keyword
                    })
                    break
        
        return comparative_sentences
    
    def handle_negation(self, text):
        """
        Maneja palabras de negación para mejorar la clasificación
        """
        try:
            words = word_tokenize(text.lower())
        except Exception as e:
            print(f"Error en tokenización: {e}")
            # Fallback a división simple por espacios
            words = text.lower().split()
        
        negated_text = []
        negate = False
        
        for word in words:
            if word in self.negation_words:
                negate = True
                negated_text.append(word)
            elif word in ['.', '!', '?', ',']:
                negate = False
                negated_text.append(word)
            else:
                if negate:
                    negated_text.append(f"NOT_{word}")
                else:
                    negated_text.append(word)
        
        return ' '.join(negated_text)
    
    def adjust_comparative_sentiment(self, text):
        """
        NUEVA MEJORA: Ajusta el texto para darle más peso a comparaciones positivas
        """
        adjusted_text = text.lower()
        
        for comparative_word, weight in self.comparative_sentiment_adjustments.items():
            if comparative_word in adjusted_text:
                if weight > 0:
                    # Para palabras positivas, repetir para darle más peso
                    repetitions = int(abs(weight))
                    boost_words = f" {comparative_word}_POSITIVE " * repetitions
                    adjusted_text = adjusted_text.replace(comparative_word, comparative_word + boost_words)
                elif weight < 0:
                    # Para palabras negativas, marcar claramente
                    repetitions = int(abs(weight))
                    boost_words = f" {comparative_word}_NEGATIVE " * repetitions  
                    adjusted_text = adjusted_text.replace(comparative_word, comparative_word + boost_words)
        
        return adjusted_text
    
    def apply_comparative_post_processing(self, text, initial_prediction, semantic_orientation):
        """
        NUEVA MEJORA: Post-procesa predicciones usando reglas comparativas específicas
        Incluye clasificación NEUTRAL para orientaciones cercanas a cero
        """
        text_lower = text.lower()
        
        # NUEVA MEJORA: Umbral para sentimiento neutral
        NEUTRAL_THRESHOLD = 2.0  # Si SO está entre -2 y +2, considerar neutral
        
        # Reglas de corrección para comparaciones positivas claras
        positive_comparative_patterns = [
            'better than', 'superior', 'improved', 'enhanced', 'upgraded',
            'more advanced', 'higher quality', 'greater'
        ]
        
        negative_comparative_patterns = [
            'worse than', 'inferior', 'declined', 'deteriorated', 
            'lower quality', 'less impressive'
        ]
        
        # Patrones neutrales específicos
        neutral_patterns = [
            'similar to', 'same as', 'comparable', 'equivalent', 'equal',
            'different but', 'neither good nor bad', 'average', 'okay', 'fine',
            'similar', 'comparable to', 'like other', 'typical'  # Añadido "similar" directo
        ]
        
        # Contar patrones
        positive_count = sum(1 for pattern in positive_comparative_patterns if pattern in text_lower)
        negative_count = sum(1 for pattern in negative_comparative_patterns if pattern in text_lower)
        neutral_count = sum(1 for pattern in neutral_patterns if pattern in text_lower)
        
        # NUEVA LÓGICA: Clasificación tripartita (Positivo/Neutral/Negativo)
        
        # 1. PRIORIDAD ALTA: Si hay patrones neutrales claros
        if neutral_count > 0:
            return 2  # NEUTRAL
        
        # 2. PRIORIDAD ALTA: Si SO está en zona neutral Y no hay predominio claro
        if abs(semantic_orientation) <= NEUTRAL_THRESHOLD:
            if positive_count <= negative_count + 1 and negative_count <= positive_count + 1:  # Empate o diferencia mínima
                return 2  # NEUTRAL
        
        # 3. Si la orientación semántica es fuertemente positiva y hay patrones comparativos positivos
        if (semantic_orientation > 10 and positive_count > negative_count and 
            any(pattern in text_lower for pattern in positive_comparative_patterns)):
            return 1  # Forzar a positivo
        
        # 4. Si la orientación semántica es fuertemente negativa y hay patrones comparativos negativos  
        elif (semantic_orientation < -10 and negative_count > positive_count and
              any(pattern in text_lower for pattern in negative_comparative_patterns)):
            return 0  # Forzar a negativo
        
        # 5. CASO ESPECIAL: "similar" sin otros indicadores fuertes = NEUTRAL
        if 'similar' in text_lower and abs(semantic_orientation) <= 5.0:
            return 2  # NEUTRAL
        
        # 6. Zona neutral por orientación semántica débil (expandida)
        elif abs(semantic_orientation) <= NEUTRAL_THRESHOLD:
            return 2  # NEUTRAL
        
        # 7. En otros casos, usar orientación semántica como guía
        if semantic_orientation > NEUTRAL_THRESHOLD:
            return 1  # Positivo
        elif semantic_orientation < -NEUTRAL_THRESHOLD:
            return 0  # Negativo
        else:
            return 2  # Neutral por defecto

def create_advanced_pipeline():
    """
    VERSIÓN RÁPIDA: Pipeline optimizado para velocidad manteniendo mejoras
    """
    # Vectorizador simplificado pero efectivo
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Solo bigramas para velocidad
        max_features=10000,  # Valor fijo optimizado
        min_df=1,           
        max_df=0.9,         
        sublinear_tf=True,
        token_pattern=r'(?u)\b\w\w+\b|_POSITIVE|_NEGATIVE'
    )
    
    # Solo SVM (más rápido que ensemble y bueno con características complejas)
    classifier = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
    
    # Pipeline simplificado
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    return pipeline

def main():
    """Función principal que ejecuta el análisis completo"""
    print("=== Sistema Avanzado de Análisis de Sentimientos ===")
    print("Implementando técnicas de Minería de Opiniones Basadas en Aspectos\n")
    
    # Inicializar el analizador
    analyzer = AspectBasedSentimentAnalyzer()
    
    # Cargar datos
    print("Cargando dataset de reseñas de películas...")
    try:
        movie_reviews = load_files(nltk.data.find('corpora/movie_reviews'))
        X, y = movie_reviews.data, movie_reviews.target
        print(f"Dataset cargado: {len(X)} documentos")
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return
    
    # Preprocesamiento avanzado
    print("\nRealizando preprocesamiento avanzado...")
    processed_documents = []
    
    for i, doc in enumerate(X[:500]):  # Reducido de 1000 a 500 para velocidad
        text = analyzer.preprocess_text(doc)
        # MEJORAS APLICADAS: Aplicar manejo de negación Y ajuste comparativo
        text_with_negation = analyzer.handle_negation(text)
        text_with_comparative_boost = analyzer.adjust_comparative_sentiment(text_with_negation)
        processed_documents.append(text_with_comparative_boost)
        
        if (i + 1) % 200 == 0:
            print(f"Procesados {i + 1} documentos...")
    
    y_subset = y[:500]
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        processed_documents, y_subset, test_size=0.2, random_state=42, stratify=y_subset
    )
    
    # Crear y entrenar pipeline avanzado
    print("\nCreando pipeline avanzado con ensemble de clasificadores...")
    pipeline = create_advanced_pipeline()
    
    # VERSIÓN ULTRA RÁPIDA: Sin Grid Search, usando parámetros pre-optimizados
    print("Entrenando modelo con parámetros optimizados...")
    pipeline.fit(X_train, y_train)
    
    print("Parámetros utilizados: C=10.0, gamma=scale, max_features=10000, ngram_range=(1,2)")
    
    # Usar pipeline directamente
    grid_search = pipeline  # Para mantener compatibilidad con el resto del código
    
    # Evaluación del modelo
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== RESULTADOS DEL MODELO ===")
    print(f"Precisión en test: {accuracy:.4f}")
    print(f"\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo']))
    print(f"\nNOTA: El post-procesamiento añade clasificación NEUTRAL para orientaciones semánticas entre -2.0 y +2.0")
    
    # Análisis de aspectos en ejemplos
    print(f"\n=== ANÁLISIS DE ASPECTOS Y COMPARACIONES ===")
    
    sample_reviews = [
        "The movie was excellent with great acting but poor sound quality",
        "This film is better than the previous one, with superior cinematography", 
        "The plot was terrible and the worst I've ever seen",
        "Similar to other movies but with different approach to storytelling",
        # NUEVOS EJEMPLOS para probar clasificación NEUTRAL
        "The movie was okay, neither particularly good nor bad",
        "It's similar to other films in the genre, nothing special",
        "The acting was fine, the plot was average, overall just okay"
    ]
    
    for i, review in enumerate(sample_reviews, 1):
        print(f"\nReseña {i}: '{review}'")
        
        # Predicción de sentimiento con MEJORAS aplicadas
        processed_review = analyzer.handle_negation(review)
        boosted_review = analyzer.adjust_comparative_sentiment(processed_review)
        initial_prediction = grid_search.predict([boosted_review])[0]
        
        # Calcular orientación semántica para post-procesamiento
        so_score = analyzer.calculate_semantic_orientation(review)
        
        # NUEVA MEJORA: Aplicar post-procesamiento comparativo CON NEUTRAL
        final_prediction = analyzer.apply_comparative_post_processing(
            review, initial_prediction, so_score
        )
        
        # Mapear códigos a etiquetas
        sentiment_labels = {0: "Negativo", 1: "Positivo", 2: "Neutral"}
        sentiment = sentiment_labels.get(final_prediction, "Positivo" if final_prediction == 1 else "Negativo")
        
        correction_applied = ""
        if final_prediction != initial_prediction:
            if final_prediction == 2:
                correction_applied = " (CLASIFICADO COMO NEUTRAL)"
            else:
                correction_applied = " (CORREGIDO)"
        
        print(f"Sentimiento general: {sentiment}{correction_applied}")
        
        # Mostrar predicción inicial para comparación
        initial_sentiment = "Positivo" if initial_prediction == 1 else "Negativo"
        print(f"Predicción inicial del modelo: {initial_sentiment}")
        
        # NUEVA INFO: Mostrar información de depuración para entender la clasificación
        if abs(so_score) <= 2.0:
            print(f"Orientación semántica en zona NEUTRAL: {so_score:.3f} (umbral: ±2.0)")
        
        # Detectar patrones para explicación
        neutral_found = any(pattern in review.lower() for pattern in [
            'similar', 'comparable', 'same as', 'equivalent', 'average', 'okay', 'fine'
        ])
        
        if neutral_found and final_prediction == 2:
            print(f"Patrones neutrales detectados en el texto")
        
        # Mostrar texto procesado para debugging
        print(f"Texto procesado: {boosted_review[:100]}...")
        
        # Extracción de aspectos
        aspects = analyzer.extract_aspects_and_opinions(review)
        if aspects:
            print(f"Aspectos encontrados: {[asp[0] for asp in aspects]}")
        
        # Detección de comparaciones
        comparisons = analyzer.detect_comparative_sentences(review)
        if comparisons:
            for comp in comparisons:
                print(f"Comparación detectada: {comp['type']} - '{comp['keyword']}'")
        
        # Orientación semántica (ya calculada arriba)
        print(f"Orientación semántica (SO): {so_score:.3f}")

if __name__ == "__main__":
    main()