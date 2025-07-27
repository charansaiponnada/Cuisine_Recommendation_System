# Updated app.py with better error handling and debugging
from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import random
from datetime import datetime
import os
import traceback

app = Flask(__name__)
app.secret_key = 'cuisine_recommender_2024'

class CuisineRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.cuisine_features = None
        self.svd_model = None
        self.nmf_model = None
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        self.cuisines_df = None
        self.users_df = None
        self.ratings_df = None
        print("🍽️ Initializing Cuisine Recommender...")
        # Initialize the recommender system
        

    def generate_synthetic_data(self):
        """Generate realistic cuisine recommendation data"""
        print("📊 Generating synthetic data...")
        
        # Define cuisines with rich metadata
        cuisines_data = [
            {"cuisine_id": 1, "name": "Italian", "description": "pasta pizza risotto cheese tomato basil olive_oil", 
             "spice_level": 2, "prep_time": 45, "difficulty": 3, "dietary_tags": "vegetarian_friendly gluten_options", 
             "region": "European", "price_range": 3, "popularity": 9.2},
            {"cuisine_id": 2, "name": "Thai", "description": "spicy coconut curry lemongrass chili lime fish_sauce", 
             "spice_level": 8, "prep_time": 35, "difficulty": 4, "dietary_tags": "gluten_free spicy", 
             "region": "Asian", "price_range": 2, "popularity": 8.7},
            {"cuisine_id": 3, "name": "Mexican", "description": "spicy beans corn avocado lime cilantro chili pepper", 
             "spice_level": 6, "prep_time": 30, "difficulty": 3, "dietary_tags": "vegetarian_options spicy", 
             "region": "Latin_American", "price_range": 2, "popularity": 8.9},
            {"cuisine_id": 4, "name": "Japanese", "description": "sushi rice soy_sauce wasabi ginger fresh clean", 
             "spice_level": 1, "prep_time": 60, "difficulty": 5, "dietary_tags": "gluten_free pescatarian", 
             "region": "Asian", "price_range": 4, "popularity": 9.1},
            {"cuisine_id": 5, "name": "Indian", "description": "spicy curry turmeric garam_masala rice naan yogurt", 
             "spice_level": 9, "prep_time": 55, "difficulty": 4, "dietary_tags": "vegetarian_options very_spicy", 
             "region": "Asian", "price_range": 2, "popularity": 8.5},
            {"cuisine_id": 6, "name": "French", "description": "butter wine cheese cream sauce elegant refined", 
             "spice_level": 2, "prep_time": 75, "difficulty": 5, "dietary_tags": "dairy_heavy sophisticated", 
             "region": "European", "price_range": 5, "popularity": 8.3},
            {"cuisine_id": 7, "name": "Chinese", "description": "soy_sauce ginger garlic stir_fry rice noodles wok", 
             "spice_level": 4, "prep_time": 25, "difficulty": 3, "dietary_tags": "quick_cooking vegetarian_options", 
             "region": "Asian", "price_range": 2, "popularity": 9.0},
            {"cuisine_id": 8, "name": "Korean", "description": "kimchi spicy fermented sesame gochujang bbq rice", 
             "spice_level": 7, "prep_time": 40, "difficulty": 4, "dietary_tags": "fermented spicy", 
             "region": "Asian", "price_range": 3, "popularity": 8.6},
            {"cuisine_id": 9, "name": "Mediterranean", "description": "olive_oil feta olives herbs fresh vegetables healthy", 
             "spice_level": 3, "prep_time": 35, "difficulty": 2, "dietary_tags": "healthy vegetarian_friendly", 
             "region": "European", "price_range": 3, "popularity": 8.8},
            {"cuisine_id": 10, "name": "American", "description": "burger fries bbq comfort_food cheese bacon hearty", 
             "spice_level": 3, "prep_time": 30, "difficulty": 2, "dietary_tags": "comfort_food hearty", 
             "region": "North_American", "price_range": 3, "popularity": 8.2}
        ]
        
        self.cuisines_df = pd.DataFrame(cuisines_data)
        # Save cuisines DataFrame to CSV
        self.cuisines_df.to_csv('cuisines.csv', index=False)
        print(f"✅ Generated {len(self.cuisines_df)} cuisines and saved to cuisines.csv")
        
        # Generate users with preferences
        user_profiles = []
        for i in range(1, 101):
            profile = {
                "user_id": i,
                "age_group": random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
                "dietary_preference": random.choice(["none", "vegetarian", "vegan", "gluten_free", "pescatarian"]),
                "spice_tolerance": random.randint(1, 10),
                "budget_preference": random.randint(1, 5),
                "adventurous_score": random.randint(1, 10),
                "region_preference": random.choice(["Asian", "European", "Latin_American", "North_American", "Mixed"])
            }
            user_profiles.append(profile)
        
        self.users_df = pd.DataFrame(user_profiles)
        # Save users DataFrame to CSV
        self.users_df.to_csv('users.csv', index=False)
        print(f"✅ Generated {len(self.users_df)} user profiles and saved to users.csv")
        
        # Generate realistic ratings based on user preferences
        ratings_data = []
        for user_id in range(1, 101):
            user = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
            
            # Each user rates 5-8 cuisines
            num_ratings = random.randint(5, 8)
            rated_cuisines = random.sample(range(1, 11), num_ratings)
            
            for cuisine_id in rated_cuisines:
                cuisine = self.cuisines_df[self.cuisines_df['cuisine_id'] == cuisine_id].iloc[0]
                
                # Base rating influenced by user preferences
                base_rating = 3.0
                
                # Spice preference alignment
                spice_diff = abs(user['spice_tolerance'] - cuisine['spice_level'])
                if spice_diff <= 2:
                    base_rating += 0.8
                elif spice_diff <= 4:
                    base_rating += 0.3
                else:
                    base_rating -= 0.5
                
                # Budget alignment
                budget_diff = abs(user['budget_preference'] - cuisine['price_range'])
                if budget_diff <= 1:
                    base_rating += 0.5
                elif budget_diff > 2:
                    base_rating -= 0.4
                
                # Adventurous score vs cuisine difficulty
                if user['adventurous_score'] >= 7 and cuisine['difficulty'] >= 4:
                    base_rating += 0.6
                elif user['adventurous_score'] <= 3 and cuisine['difficulty'] <= 2:
                    base_rating += 0.4
                
                # Region preference
                if user['region_preference'] == cuisine['region'] or user['region_preference'] == "Mixed":
                    base_rating += 0.3
                
                # Add some randomness
                base_rating += random.uniform(-0.8, 0.8)
                
                # Ensure rating is between 1 and 5
                rating = max(1.0, min(5.0, base_rating))
                
                ratings_data.append({
                    "user_id": user_id,
                    "cuisine_id": cuisine_id,
                    "rating": round(rating, 1),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        self.ratings_df = pd.DataFrame(ratings_data)
        # Save ratings DataFrame to CSV
        self.ratings_df.to_csv('ratings.csv', index=False)
        print(f"✅ Generated {len(self.ratings_df)} ratings and saved to ratings.csv")
        
    def build_collaborative_filtering(self):
        """Build collaborative filtering models"""
        try:
            print("🤖 Building collaborative filtering models...")
            
            # Create user-item matrix
            self.user_item_matrix = self.ratings_df.pivot(
                index='user_id', 
                columns='cuisine_id', 
                values='rating'
            ).fillna(0)
            
            print(f"✅ User-item matrix shape: {self.user_item_matrix.shape}")
            
            # SVD Model
            self.svd_model = TruncatedSVD(n_components=5, random_state=42)
            user_factors = self.svd_model.fit_transform(self.user_item_matrix)
            print("✅ SVD model trained")
            
            # NMF Model
            self.nmf_model = NMF(n_components=5, random_state=42, max_iter=1000)
            self.nmf_model.fit(self.user_item_matrix)
            print("✅ NMF model trained")
            
        except Exception as e:
            print(f"❌ Error in collaborative filtering: {str(e)}")
            raise
        
    def build_content_based_filtering(self):
        """Build content-based filtering using TF-IDF"""
        try:
            print("📝 Building content-based filtering...")
            
            # Combine all text features
            self.cuisines_df['combined_features'] = (
                self.cuisines_df['description'] + ' ' +
                self.cuisines_df['dietary_tags'] + ' ' +
                self.cuisines_df['region'] + ' ' +
                self.cuisines_df['spice_level'].astype(str) + '_spice ' +
                self.cuisines_df['difficulty'].astype(str) + '_difficulty'
            )
            
            # TF-IDF Vectorization
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.cuisines_df['combined_features'])
            
            # Compute similarity matrix
            self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
            print("✅ Content-based model built")
            
        except Exception as e:
            print(f"❌ Error in content-based filtering: {str(e)}")
            raise
        
    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        """Get recommendations using collaborative filtering"""
        try:
            if user_id not in self.user_item_matrix.index:
                print(f"⚠️ User {user_id} not found, returning popular recommendations")
                return self.get_popular_recommendations(n_recommendations)
            
            user_ratings = self.user_item_matrix.loc[user_id]
            
            # SVD predictions
            user_idx = list(self.user_item_matrix.index).index(user_id)
            user_factors = self.svd_model.transform(self.user_item_matrix.iloc[[user_idx]])
            predicted_ratings = user_factors.dot(self.svd_model.components_)
            
            recommendations = []
            for i, cuisine_id in enumerate(self.user_item_matrix.columns):
                if user_ratings[cuisine_id] == 0:  # Not rated by user
                    predicted_rating = predicted_ratings[0][i]
                    recommendations.append((cuisine_id, predicted_rating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"❌ Error in collaborative recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations)
    
    def get_content_based_recommendations(self, user_id, n_recommendations=5):
        """Get recommendations using content-based filtering"""
        try:
            if user_id not in self.user_item_matrix.index:
                return self.get_popular_recommendations(n_recommendations)
            
            user_ratings = self.user_item_matrix.loc[user_id]
            rated_cuisines = user_ratings[user_ratings > 0]
            
            if len(rated_cuisines) == 0:
                return self.get_popular_recommendations(n_recommendations)
            
            # Find cuisines similar to highly rated ones
            recommendations = {}
            for cuisine_id, rating in rated_cuisines.items():
                if rating >= 4.0:  # Only consider highly rated cuisines
                    cuisine_idx = cuisine_id - 1  # Convert to 0-based index
                    similar_scores = self.content_similarity_matrix[cuisine_idx]
                    
                    for i, score in enumerate(similar_scores):
                        similar_cuisine_id = i + 1  # Convert back to 1-based
                        if similar_cuisine_id != cuisine_id and user_ratings[similar_cuisine_id] == 0:
                            if similar_cuisine_id not in recommendations:
                                recommendations[similar_cuisine_id] = 0
                            recommendations[similar_cuisine_id] += score * rating
            
            if not recommendations:
                return self.get_popular_recommendations(n_recommendations)
            
            sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return sorted_recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"❌ Error in content-based recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations)
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=5):
        """Combine collaborative and content-based recommendations"""
        try:
            collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
            content_recs = self.get_content_based_recommendations(user_id, n_recommendations * 2)
            
            # Combine and weight the recommendations
            hybrid_scores = {}
            
            # Weight collaborative filtering (60%)
            for cuisine_id, score in collab_recs:
                hybrid_scores[cuisine_id] = score * 0.6
            
            # Weight content-based filtering (40%)
            for cuisine_id, score in content_recs:
                if cuisine_id in hybrid_scores:
                    hybrid_scores[cuisine_id] += score * 0.4
                else:
                    hybrid_scores[cuisine_id] = score * 0.4
            
            sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"❌ Error in hybrid recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations)
    
    def get_popular_recommendations(self, n_recommendations=5):
        """Get popular cuisines for new users"""
        try:
            popular_cuisines = self.cuisines_df.nlargest(n_recommendations, 'popularity')
            return [(row['cuisine_id'], row['popularity']) for _, row in popular_cuisines.iterrows()]
        except Exception as e:
            print(f"❌ Error in popular recommendations: {str(e)}")
            return [(1, 5.0), (2, 4.5), (3, 4.0), (4, 3.5), (5, 3.0)]  # Fallback
    
    def get_mood_based_recommendations(self, mood, n_recommendations=5):
        """Get recommendations based on user's current mood"""
        try:
            if mood == "adventurous":
                filtered_cuisines = self.cuisines_df[self.cuisines_df['difficulty'] >= 4]
            elif mood == "comfort":
                filtered_cuisines = self.cuisines_df[
                    (self.cuisines_df['spice_level'] <= 4) & 
                    (self.cuisines_df['difficulty'] <= 3)
                ]
            elif mood == "spicy":
                filtered_cuisines = self.cuisines_df[self.cuisines_df['spice_level'] >= 6]
            elif mood == "healthy":
                filtered_cuisines = self.cuisines_df[
                    self.cuisines_df['dietary_tags'].str.contains('healthy|vegetarian', case=False, na=False)
                ]
            else:
                filtered_cuisines = self.cuisines_df
            
            if len(filtered_cuisines) == 0:
                filtered_cuisines = self.cuisines_df
            
            top_cuisines = filtered_cuisines.nlargest(n_recommendations, 'popularity')
            return [(row['cuisine_id'], row['popularity']) for _, row in top_cuisines.iterrows()]
            
        except Exception as e:
            print(f"❌ Error in mood-based recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations)

# Initialize the recommender system
print("🚀 Starting Cuisine Recommender System...")
recommender = CuisineRecommender()
recommender.generate_synthetic_data()
recommender.build_collaborative_filtering()
recommender.build_content_based_filtering()
print("✅ System initialization complete!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    try:
        # Get some statistics for the dashboard
        total_users = len(recommender.users_df)
        total_cuisines = len(recommender.cuisines_df)
        total_ratings = len(recommender.ratings_df)
        avg_rating = recommender.ratings_df['rating'].mean()
        
        stats = {
            'total_users': total_users,
            'total_cuisines': total_cuisines,
            'total_ratings': total_ratings,
            'avg_rating': round(avg_rating, 2)
        }
        
        return render_template('dashboard.html', stats=stats)
    except Exception as e:
        print(f"❌ Error in dashboard: {str(e)}")
        return render_template('dashboard.html', stats={'total_users': 0, 'total_cuisines': 0, 'total_ratings': 0, 'avg_rating': 0})

@app.route('/api/cuisines')
def get_cuisines():
    try:
        cuisines = recommender.cuisines_df.to_dict('records')
        return jsonify(cuisines)
    except Exception as e:
        print(f"❌ Error in get_cuisines: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend/<int:user_id>')
def get_recommendations(user_id):
    try:
        method = request.args.get('method', 'hybrid')
        print(f"🔍 Getting {method} recommendations for user {user_id}")
        
        if method == 'collaborative':
            recs = recommender.get_collaborative_recommendations(user_id)
        elif method == 'content':
            recs = recommender.get_content_based_recommendations(user_id)
        elif method == 'popular':
            recs = recommender.get_popular_recommendations()
        else:
            recs = recommender.get_hybrid_recommendations(user_id)
        
        # Get cuisine details for recommendations
        recommendations = []
        for cuisine_id, score in recs:
            try:
                cuisine = recommender.cuisines_df[recommender.cuisines_df['cuisine_id'] == cuisine_id].iloc[0]
                recommendations.append({
                    'cuisine_id': int(cuisine_id),
                    'name': cuisine['name'],
                    'description': cuisine['description'],
                    'spice_level': int(cuisine['spice_level']),
                    'prep_time': int(cuisine['prep_time']),
                    'difficulty': int(cuisine['difficulty']),
                    'price_range': int(cuisine['price_range']),
                    'region': cuisine['region'],
                    'score': round(float(score), 2)
                })
            except Exception as e:
                print(f"⚠️ Error processing cuisine {cuisine_id}: {str(e)}")
                continue
        
        print(f"✅ Returning {len(recommendations)} recommendations")
        return jsonify(recommendations)
        
    except Exception as e:
        print(f"❌ Error in get_recommendations: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mood_recommend')
def get_mood_recommendations():
    try:
        mood = request.args.get('mood', 'comfort')
        print(f"🎭 Getting {mood} mood recommendations")
        
        recs = recommender.get_mood_based_recommendations(mood)
        
        recommendations = []
        for cuisine_id, score in recs:
            try:
                cuisine = recommender.cuisines_df[recommender.cuisines_df['cuisine_id'] == cuisine_id].iloc[0]
                recommendations.append({
                    'cuisine_id': int(cuisine_id),
                    'name': cuisine['name'],
                    'description': cuisine['description'],
                    'spice_level': int(cuisine['spice_level']),
                    'prep_time': int(cuisine['prep_time']),
                    'difficulty': int(cuisine['difficulty']),
                    'price_range': int(cuisine['price_range']),
                    'region': cuisine['region'],
                    'score': round(float(score), 2)
                })
            except Exception as e:
                print(f"⚠️ Error processing cuisine {cuisine_id}: {str(e)}")
                continue
        
        print(f"✅ Returning {len(recommendations)} mood recommendations")
        return jsonify(recommendations)
        
    except Exception as e:
        print(f"❌ Error in mood recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rate', methods=['POST'])
def rate_cuisine():
    try:
        data = request.json
        user_id = data['user_id']
        cuisine_id = data['cuisine_id']
        rating = data['rating']
        
        # Add the rating to our dataset
        new_rating = {
            'user_id': user_id,
            'cuisine_id': cuisine_id,
            'rating': rating,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        recommender.ratings_df = pd.concat([
            recommender.ratings_df, 
            pd.DataFrame([new_rating])
        ], ignore_index=True)
        
        # Rebuild the user-item matrix
        recommender.user_item_matrix = recommender.ratings_df.pivot(
            index='user_id', 
            columns='cuisine_id', 
            values='rating'
        ).fillna(0)
        
        return jsonify({'status': 'success', 'message': 'Rating added successfully'})
        
    except Exception as e:
        print(f"❌ Error in rate_cuisine: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<int:user_id>')
def get_user_profile(user_id):
    try:
        if user_id in recommender.users_df['user_id'].values:
            user = recommender.users_df[recommender.users_df['user_id'] == user_id].iloc[0].to_dict()
            user_ratings = recommender.ratings_df[recommender.ratings_df['user_id'] == user_id]
            user['total_ratings'] = len(user_ratings)
            user['avg_rating'] = round(user_ratings['rating'].mean(), 2) if len(user_ratings) > 0 else 0
            return jsonify(user)
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        print(f"❌ Error in get_user_profile: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_system_stats():
    try:
        # Calculate various statistics
        stats = {
            'total_users': len(recommender.users_df),
            'total_cuisines': len(recommender.cuisines_df),
            'total_ratings': len(recommender.ratings_df),
            'avg_rating': round(recommender.ratings_df['rating'].mean(), 2),
            'most_popular_cuisine': recommender.cuisines_df.loc[recommender.cuisines_df['popularity'].idxmax()]['name'],
            'spiciest_cuisine': recommender.cuisines_df.loc[recommender.cuisines_df['spice_level'].idxmax()]['name'],
            'rating_distribution': recommender.ratings_df['rating'].value_counts().sort_index().to_dict()
        }
        
        return jsonify(stats)
    except Exception as e:
        print(f"❌ Error in get_system_stats: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("🌟 Starting Flask server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)