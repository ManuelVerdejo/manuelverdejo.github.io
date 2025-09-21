import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split, RandomizedSearchCV
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class DataManager:
    """Maneja carga única de datos y mapeo correcto de IDs"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.dataset = None
        self.trainset = None
        self.testset = None
        self.raw_to_inner_id = {}
        self.inner_to_raw_id = {}
        self.item_metadata = {}
        self.test_size = test_size
        self.random_state = random_state
        
    def load_data_once(self):
        """CORREGIDO: Carga datos una sola vez y mapea IDs correctamente"""
        print("Cargando dataset una sola vez...")
        
        # Cargar dataset principal
        self.dataset = Dataset.load_builtin('ml-100k')
        self.trainset, self.testset = train_test_split(
            self.dataset, test_size=self.test_size, random_state=self.random_state
        )
        
        # CORRECCIÓN CRÍTICA: Crear mapeos correctos entre IDs internos y externos
        self._create_id_mappings()
        self._create_realistic_metadata()
        
        print(f"Datos cargados: {self.trainset.n_users} usuarios, {self.trainset.n_items} items")
        return self.dataset, self.trainset, self.testset
    
    def _create_id_mappings(self):
        """Crea mapeos correctos entre IDs internos de Surprise y IDs reales"""
        # CORRECCIÓN: Desempaquetar los 4 valores para evitar el error
        for user_raw_id, item_raw_id, rating, timestamp in self.dataset.raw_ratings:
            if item_raw_id not in self.raw_to_inner_id:
                try:
                    inner_id = self.trainset.to_inner_iid(item_raw_id)
                    self.raw_to_inner_id[item_raw_id] = inner_id
                    self.inner_to_raw_id[inner_id] = item_raw_id
                except ValueError:
                    # Este ítem no está en el trainset, puede estar en el testset
                    continue
    
    def _create_realistic_metadata(self):
        """CORRIGIDO: Asigna metadata usando IDs correctos del dataset"""
        np.random.seed(42)
        genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Sci-Fi', 'Horror', 'Adventure']
        
        # Usar IDs RAW del dataset, no IDs internos de Surprise
        all_raw_items = set(raw_id for _, raw_id, _, _ in self.dataset.raw_ratings)
        
        for raw_item_id in all_raw_items:
            n_genres = np.random.randint(1, 4)
            selected_genres = np.random.choice(genres, n_genres, replace=False).tolist()
            
            # Simular año de lanzamiento y rating promedio
            release_year = np.random.randint(1980, 2024)
            avg_rating = np.random.uniform(2.0, 4.5)
            
            self.item_metadata[raw_item_id] = {
                'genres': selected_genres,
                'release_year': release_year,
                'avg_rating': avg_rating
            }
    
    def get_item_metadata(self, raw_item_id):
        """Obtiene metadata usando ID real del item"""
        return self.item_metadata.get(raw_item_id, {'genres': ['Unknown'], 'release_year': 2000, 'avg_rating': 3.0})

class ModelTrainer:
    """Responsabilidad única: entrenar y optimizar modelos"""
    
    def __init__(self, n_iter=10, cv=3, n_jobs=-1, random_state=42):
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = None
    
    def train_optimized_model(self, dataset):
        """Entrena modelo con parámetros flexibles"""
        print(f"Entrenando con RandomizedSearch (n_iter={self.n_iter}, cv={self.cv})...")
        
        param_distributions = {
            'n_factors': [50, 100, 150],
            'lr_all': [0.005, 0.01, 0.02],
            'reg_all': [0.02, 0.1, 0.2],
            'n_epochs': [20, 30]
        }
        
        rs = RandomizedSearchCV(
            SVD, 
            param_distributions, 
            measures=['rmse'], 
            cv=self.cv,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        rs.fit(dataset)
        
        self.best_model = rs.best_estimator['rmse']
        self.best_params = rs.best_params['rmse']
        self.best_score = rs.best_score['rmse']
        
        print(f"Mejor RMSE: {self.best_score:.4f}")
        print(f"Mejores parámetros: {self.best_params}")
        
        return self.best_model

class RecommendationEngine:
    """Responsabilidad única: generar recomendaciones"""
    
    def __init__(self, model, data_manager, max_candidates=None):
        self.model = model
        self.data_manager = data_manager
        self.max_candidates = max_candidates
        self.popularity_scores = {}
        self._calculate_popularity()
    
    def _calculate_popularity(self):
        """Calcula popularidad para cold start"""
        item_stats = {}
        for (user, item, rating) in self.data_manager.trainset.all_ratings():
            raw_item_id = self.data_manager.inner_to_raw_id.get(item)
            if raw_item_id not in item_stats:
                item_stats[raw_item_id] = {'sum': 0, 'count': 0}
            item_stats[raw_item_id]['sum'] += rating
            item_stats[raw_item_id]['count'] += 1
        
        for raw_item_id, stats in item_stats.items():
            avg_rating = stats['sum'] / stats['count']
            count = stats['count']
            self.popularity_scores[raw_item_id] = avg_rating * np.log1p(count)
    
    def get_recommendations(self, user_id, n_recommendations=10, diversity_threshold=0.3):
        """CORRIGIDO: Busca en TODOS los candidatos, no solo 200"""
        try:
            # Obtener items que el usuario NO ha calificado
            user_items = set()
            for (u, i, r) in self.data_manager.trainset.all_ratings():
                if u == user_id:
                    raw_item_id = self.data_manager.inner_to_raw_id.get(i)
                    if raw_item_id:
                        user_items.add(raw_item_id)
            
            all_items = set(self.data_manager.raw_to_inner_id.keys())
            candidates = list(all_items - user_items)
            
            if not candidates:
                return self._get_popular_items_for_cold_start(n_recommendations)
            
            # CORRECCIÓN: Evaluar TODOS los candidatos si max_candidates no está definido
            if self.max_candidates:
                candidates = candidates[:self.max_candidates]
            
            # Generar predicciones
            predictions = []
            for raw_item_id in candidates:
                try:
                    pred = self.model.predict(user_id, raw_item_id, verbose=False).est
                    predictions.append((raw_item_id, pred))
                except:
                    continue
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Aplicar diversidad REAL en toda la lista
            return self._apply_comprehensive_diversity(predictions, n_recommendations, diversity_threshold)
            
        except Exception as e:
            print(f"Error generando recomendaciones para usuario {user_id}: {e}")
            return self._get_popular_items_for_cold_start(n_recommendations)
    
    def _apply_comprehensive_diversity(self, predictions, n_recs, threshold):
        """CORRIGIDO: Verifica diversidad contra TODA la lista seleccionada"""
        selected = []
        
        for raw_item_id, score in predictions:
            if len(selected) >= n_recs:
                break
            
            # Verificar diversidad contra TODOS los items ya seleccionados
            is_diverse = True
            current_metadata = self.data_manager.get_item_metadata(raw_item_id)
            current_genres = set(current_metadata['genres'])
            
            for selected_item_id, _ in selected:
                selected_metadata = self.data_manager.get_item_metadata(selected_item_id)
                selected_genres = set(selected_metadata['genres'])
                
                # Calcular overlap de géneros
                if current_genres and selected_genres:
                    overlap = len(current_genres & selected_genres) / len(current_genres | selected_genres)
                    if overlap > threshold:
                        is_diverse = False
                        break
            
            if is_diverse:
                selected.append((raw_item_id, score))
        
        return selected
    
    def _get_popular_items_for_cold_start(self, n_items):
        """Manejo de cold start mejorado"""
        sorted_items = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
        return [(item_id, score) for item_id, score in sorted_items[:n_items]]
    
    def recommend_new_items(self, n_recommendations=10):
        """CORRIGIDO: Estrategia de cold start para ITEMS nuevos"""
        # Simular items nuevos (últimos años)
        recent_items = []
        for raw_item_id, metadata in self.data_manager.item_metadata.items():
            if metadata['release_year'] >= 2020:  # Items "nuevos"
                recent_items.append((raw_item_id, metadata['avg_rating']))
        
        recent_items.sort(key=lambda x: x[1], reverse=True)
        return recent_items[:n_recommendations]

class AdvancedEvaluator:
    """Responsabilidad única: evaluación con métricas corregidas"""
    
    @staticmethod
    def calculate_ndcg_at_k(predictions, k=10, threshold=3.5):
        """CORRIGIDO: nDCG usando calificaciones reales, no binarias"""
        user_est_true = {}
        for uid, _, true_r, est, _ in predictions:
            if uid not in user_est_true:
                user_est_true[uid] = []
            user_est_true[uid].append((est, true_r))
        
        ndcg_scores = []
        
        for uid, user_ratings in user_est_true.items():
            if len(user_ratings) < k:
                continue
            
            # Ordenar por predicción estimada
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            
            # CORRECCIÓN: Usar calificación real como ganancia
            dcg = 0
            for i, (est, true_r) in enumerate(user_ratings[:k]):
                gain = max(0, true_r - 1)
                dcg += gain / np.log2(i + 2)
            
            # IDCG con ratings reales ordenados
            ideal_ratings = sorted([true_r for _, true_r in user_ratings], reverse=True)
            idcg = 0
            for i, true_r in enumerate(ideal_ratings[:k]):
                gain = max(0, true_r - 1)
                idcg += gain / np.log2(i + 2)
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0
    
    @staticmethod
    def calculate_precision_recall(predictions, k=10, threshold=3.5):
        """Precision y Recall estándar"""
        user_est_true = {}
        for uid, _, true_r, est, _ in predictions:
            if uid not in user_est_true:
                user_est_true[uid] = []
            user_est_true[uid].append((est, true_r))
        
        precisions, recalls = [], []
        
        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            top_k = user_ratings[:k]
            
            relevant_items = sum(1 for (_, true_r) in user_ratings if true_r >= threshold)
            recommended_relevant = sum(1 for (_, true_r) in top_k if true_r >= threshold)
            
            if k > 0:
                precisions.append(recommended_relevant / k)
            if relevant_items > 0:
                recalls.append(recommended_relevant / relevant_items)
        
        return np.mean(precisions), np.mean(recalls)


class RecommendationSystemCoordinator:
    """CORRIGIDO: Clase coordinadora con responsabilidad única"""
    
    def __init__(self, test_size=0.2, n_iter=10, cv=3, max_candidates=None):
        # Parámetros flexibles
        self.test_size = test_size
        self.n_iter = n_iter
        self.cv = cv
        self.max_candidates = max_candidates
        
        # Componentes
        self.data_manager = None
        self.model_trainer = None
        self.recommendation_engine = None
        self.evaluator = AdvancedEvaluator()
        
    def setup_system(self):
        """Configuración completa del sistema"""
        print("=== SISTEMA CORREGIDO DE RECOMENDACIONES ===")
        
        # 1. Carga única de datos
        self.data_manager = DataManager(self.test_size)
        dataset, trainset, testset = self.data_manager.load_data_once()
        
        # 2. Entrenamiento con parámetros flexibles
        self.model_trainer = ModelTrainer(n_iter=self.n_iter, cv=self.cv)
        model = self.model_trainer.train_optimized_model(dataset)
        model.fit(trainset)
        
        # 3. Motor de recomendaciones
        self.recommendation_engine = RecommendationEngine(
            model, self.data_manager, self.max_candidates
        )
        
        return model, testset
    
    def comprehensive_evaluation(self, model, testset):
        """Evaluación completa con métricas corregidas"""
        print("\n=== EVALUACIÓN CORREGIDA ===")
        
        predictions = model.test(testset)
        
        # RMSE
        rmse = accuracy.rmse(predictions, verbose=False)
        print(f"RMSE: {rmse:.4f}")
        
        # nDCG corregido
        ndcg = self.evaluator.calculate_ndcg_at_k(predictions, k=10)
        print(f"nDCG@10 (corregido): {ndcg:.4f}")
        
        # Precision/Recall
        precision, recall = self.evaluator.calculate_precision_recall(predictions)
        print(f"Precision@10: {precision:.4f}")
        print(f"Recall@10: {recall:.4f}")
    
    def demonstrate_fixed_system(self):
        """Demostración del sistema corregido"""
        print("\n=== DEMOSTRACIÓN CORREGIDA ===")
        
        # Usuario existente
        user_id = '200' # Usar raw ID
        recs = self.recommendation_engine.get_recommendations(user_id, n_recommendations=10)
        print(f"\nRecomendaciones para usuario {user_id}:")
        for i, (item_id, score) in enumerate(recs, 1):
            metadata = self.data_manager.get_item_metadata(item_id)
            genres = ", ".join(metadata['genres'])
            print(f"  {i}. Item {item_id} (Score: {score:.3f}) - {genres} ({metadata['release_year']})")
        
        # Cold start para items nuevos
        new_items = self.recommendation_engine.recommend_new_items(5)
        print("\nItems nuevos recomendados (Cold Start):")
        for i, (item_id, rating) in enumerate(new_items, 1):
            metadata = self.data_manager.get_item_metadata(item_id)
            genres = ", ".join(metadata['genres'])
            print(f"  {i}. Item {item_id} (Rating: {rating:.3f}) - {genres} ({metadata['release_year']})")

def main():
    """Sistema corregido que aborda todos los problemas identificados"""
    
    # Crear coordinador con parámetros flexibles
    coordinator = RecommendationSystemCoordinator(
        test_size=0.2,
        n_iter=8, 
        cv=2,   
        max_candidates=None 
    )
    
    # Configurar sistema
    model, testset = coordinator.setup_system()
    
    # Evaluar con métricas corregidas
    coordinator.comprehensive_evaluation(model, testset)
    
    # Demostrar funcionalidades
    coordinator.demonstrate_fixed_system()
    
    print("\n=== SISTEMA CORREGIDO COMPLETADO ===")

if __name__ == "__main__":
    main()