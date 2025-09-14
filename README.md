> This is Second project in my intenship as Data Associate L1 in DataScience and ML Domain .
> This project was developed during my 2nd month in my 3-month internship at Infotact Solutions as a Data Science and Machine Learning Intern.
> It was my first end-to-end AI project, where I applied NLP and ML techniques to build a fully functional Cuisine Recommendation System.


# requirements.txt
Flask==2.3.3
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
Werkzeug==2.3.7
Jinja2==3.1.2

# README.md
# 🍽️ Cuisine Recommender System

A comprehensive machine learning-based recommendation engine that provides personalized cuisine suggestions to users based on their historical interactions, preferences, and behavioral patterns.

## 🚀 Features

### Core Recommendation Algorithms
- **Collaborative Filtering**: User-based and item-based recommendations using SVD and NMF
- **Content-Based Filtering**: TF-IDF vectorization of cuisine metadata and features
- **Hybrid System**: Combines collaborative and content-based methods for optimal results
- **Mood-Based Recommendations**: Context-aware suggestions based on user's current mood

### Interactive Web Interface
- **Beautiful UI**: Modern, responsive design with gradient backgrounds and smooth animations
- **User Profiles**: Detailed user preference management with 100 pre-generated profiles
- **Real-time Ratings**: Interactive star rating system with instant feedback
- **Analytics Dashboard**: Comprehensive system statistics and performance metrics
- **Multiple Recommendation Views**: Switch between different recommendation algorithms

### Advanced Features
- **Rich Cuisine Metadata**: Spice levels, preparation time, difficulty, dietary tags, and regional information
- **Contextual Filtering**: Recommendations based on mood, dietary preferences, and spice tolerance
- **Performance Metrics**: RMSE, MAE, and Precision@K evaluation
- **Real-time Updates**: Dynamic recommendation updates based on new user ratings

## 📊 Dataset

The system uses synthetically generated but realistic data including:

### User Profiles (100 users)
- Age groups, dietary preferences, spice tolerance
- Budget preferences, adventurous scores
- Regional cuisine preferences

### Cuisine Database (10 cuisines)
- **Italian**: Classic European comfort food with moderate spice
- **Thai**: Spicy Asian cuisine with coconut and lemongrass
- **Mexican**: Latin American with moderate to high spice levels
- **Japanese**: Clean, fresh Asian cuisine with minimal spice
- **Indian**: Very spicy South Asian with rich flavors
- **French**: Sophisticated European with dairy-heavy dishes
- **Chinese**: Quick-cooking Asian with moderate spice
- **Korean**: Fermented and spicy East Asian cuisine
- **Mediterranean**: Healthy European with olive oil and herbs
- **American**: Comfort food with hearty portions

### Rating System
- 1000+ realistic user-cuisine interactions
- Ratings influenced by user preferences (spice tolerance, budget, adventurous score)
- Temporal data with timestamps

## 🛠️ Technical Implementation

### Backend (Flask)
```python
# Core Components
- CuisineRecommender class with multiple ML models
- RESTful API endpoints for all functionality
- Real-time data processing and model updates
- Comprehensive error handling and validation
```

### Machine Learning Models
- **SVD (Singular Value Decomposition)**: Matrix factorization for collaborative filtering
- **NMF (Non-negative Matrix Factorization)**: Alternative matrix factorization approach
- **TF-IDF Vectorization**: Content-based similarity using cuisine features
- **Cosine Similarity**: Content-based recommendation scoring

### Frontend
- **HTML5/CSS3**: Modern responsive design with Tailwind CSS
- **JavaScript**: Interactive UI with real-time API calls
- **Chart.js**: Beautiful data visualizations and analytics
- **Font Awesome**: Professional icons and visual elements

## 📁 Project Structure

```
cuisine-recommender/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Main recommendation interface
│   └── dashboard.html    # Analytics dashboard
└── static/              # Static files (created automatically)
    ├── css/
    ├── js/
    └── images/
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone and Setup
```bash
# Create project directory
mkdir cuisine_recommender
cd cuisine_recommender

# Save the app.py file and create templates directory
mkdir templates

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Save Files
1. Save the main application code as `app.py`
2. Create the `templates` directory
3. Save the HTML templates in the templates directory:
   - `base.html`
   - `index.html` 
   - `dashboard.html`

### Step 3: Run the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

## 🎯 Usage Guide

### 1. Main Interface (`/`)
- **Select User**: Choose from 100 pre-generated user profiles (ID 1-100)
- **View Profile**: See user preferences, spice tolerance, and rating history
- **Mood Selection**: Get recommendations based on current mood:
  - Adventurous: Complex, challenging cuisines
  - Comfort Food: Familiar, low-spice options
  - Spicy: High spice level cuisines
  - Healthy: Vegetarian-friendly, nutritious options

### 2. Recommendation Methods
- **Hybrid (Recommended)**: Best of both collaborative and content-based (60/40 weight)
- **Collaborative**: Recommendations based on similar users' preferences
- **Content-Based**: Recommendations based on cuisine feature similarity
- **Popular**: Top-rated cuisines for new users

### 3. Interactive Rating
- Click "Rate This Cuisine" on any recommendation
- Provide 1-5 star ratings
- System updates recommendations in real-time

### 4. Analytics Dashboard (`/dashboard`)
- System performance metrics
- Rating distribution charts
- Cuisine popularity analysis
- Live statistics with auto-refresh

## 🔍 API Endpoints

### Core Endpoints
- `GET /` - Main recommendation interface
- `GET /dashboard` - Analytics dashboard
- `GET /api/cuisines` - Get all cuisines with metadata
- `GET /api/recommend/<user_id>?method=<method>` - Get recommendations
- `GET /api/mood_recommend?mood=<mood>` - Mood-based recommendations
- `POST /api/rate` - Submit user rating
- `GET /api/user/<user_id>` - Get user profile
- `GET /api/stats` - System statistics

### Example API Usage
```javascript
// Get hybrid recommendations for user 5
fetch('/api/recommend/5?method=hybrid')
  .then(response => response.json())
  .then(data => console.log(data));

// Submit a rating
fetch('/api/rate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    user_id: 5,
    cuisine_id: 2,
    rating: 4.5
  })
});
```

## 🧪 Week-by-Week Implementation

### Week 1: Data Collection & EDA ✅
- [x] Generate synthetic but realistic cuisine interaction dataset
- [x] Implement data preprocessing and cleaning
- [x] Create comprehensive EDA with user-item interaction analysis
- [x] Build user profiles with preferences and dietary restrictions

### Week 2: Collaborative Filtering ✅
- [x] Implement user-based collaborative filtering
- [x] Build matrix factorization models (SVD, NMF)
- [x] Evaluate models using RMSE, MAE, and Precision@K
- [x] Create user-item interaction matrices

### Week 3: Content-Based & Hybrid ✅
- [x] Build content-based system using TF-IDF on cuisine metadata
- [x] Implement hybrid recommendation system
- [x] Add user feedback loops and real-time rating updates
- [x] Create mood-based contextual recommendations

### Week 4: Interface & Evaluation ✅
- [x] Develop beautiful web interface with Flask
- [x] Create interactive dashboard with analytics
- [x] Implement comprehensive model comparison
- [x] Add real-time recommendation updates

## 📈 Model Performance

### Evaluation Metrics
- **RMSE**: Root Mean Square Error for rating prediction accuracy
- **MAE**: Mean Absolute Error for average prediction error
- **Precision@K**: Precision of top-K recommendations
- **Coverage**: Percentage of items that can be recommended
- **Diversity**: Average dissimilarity of recommended items

### Expected Performance
- **Hybrid Model**: Best overall performance combining collaborative and content-based strengths
- **Collaborative Filtering**: Excellent for users with sufficient rating history
- **Content-Based**: Great for new users and niche preferences
- **Mood-Based**: High user satisfaction for contextual recommendations

## 🎨 UI/UX Features

### Visual Design
- **Gradient Backgrounds**: Modern purple-blue gradients throughout
- **Hover Effects**: Smooth transitions and interactive elements
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Icon Integration**: Font Awesome icons for intuitive navigation

### User Experience
- **Intuitive Navigation**: Clear navigation between home and dashboard
- **Real-time Feedback**: Instant visual feedback for all interactions
- **Loading States**: Smooth loading animations and state management
- **Error Handling**: Graceful error messages and fallback options

## 🚀 Future Enhancements

### Advanced Features
- **Social Integration**: Friend recommendations and social sharing
- **Seasonal Recommendations**: Weather and season-based suggestions
- **Recipe Integration**: Full recipe details and cooking instructions
- **Restaurant Finder**: Integration with restaurant APIs and location services

### Technical Improvements
- **Deep Learning**: Neural collaborative filtering and deep content models
- **Real-time Streaming**: Kafka integration for real-time data processing
- **A/B Testing**: Framework for testing different recommendation strategies
- **Scalability**: Redis caching and database optimization

## 📝 License

This project is created for educational purposes and demonstrates a complete ML recommendation system implementation.

## 🤝 Contributing

This is a educational project, but suggestions and improvements are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

---

**Built with ❤️ using Flask, scikit-learn, and modern web technologies**
