# AI Calorie Tracking System - Backend

This is the Django backend for the AI-powered calorie tracking system that uses computer vision and machine learning to identify food items from images and estimate their nutritional values.

## Features

- **User Authentication**: Registration, login, and profile management
- **Food Detection**: YOLOv8-based image analysis for food recognition
- **Nutrition Tracking**: Daily calorie and macronutrient tracking
- **BMI & TDEE Calculation**: Automatic calculation of Basal Metabolic Rate and Total Daily Energy Expenditure
- **Progress Visualization**: Weekly reports and nutrition summaries
- **RESTful API**: Complete API for frontend integration

## Tech Stack

- **Framework**: Django 5.2.4
- **API**: Django REST Framework
- **AI/ML**: YOLOv8 (Ultralytics), PyTorch, OpenCV
- **Database**: SQLite (development), PostgreSQL (production ready)
- **Authentication**: Token-based authentication
- **CORS**: django-cors-headers for frontend integration

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   .\venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create superuser**
   ```bash
   python manage.py createsuperuser
   ```

7. **Run the development server**
   ```bash
   python manage.py runserver
   ```

The backend will be available at `http://localhost:8000`

## API Endpoints

### Authentication
- `POST /api/auth/register/` - User registration
- `POST /api/auth/login/` - User login
- `GET /api/auth/logout/` - User logout

### User Profile
- `GET /api/profile/` - Get user profile
- `POST /api/profile/` - Create/update user profile

### Food Detection
- `POST /api/detect-food/` - Upload image for food detection
- `GET /api/detection-history/` - Get detection history

### Nutrition Tracking
- `GET /api/nutrition-log/` - Get daily nutrition logs
- `POST /api/nutrition-log/` - Add nutrition log entry
- `GET /api/nutrition-summary/` - Get daily nutrition summary
- `GET /api/weekly-report/` - Get weekly nutrition report

### Food Items
- `GET /api/food-items/` - List all food items
- `POST /api/food-items/` - Add new food item

## Models

### UserProfile
- User demographic and fitness data
- Automatic BMI, BMR, and TDEE calculation
- Personalized nutrition goals

### FoodItem
- Comprehensive nutrition database
- Calories, protein, carbs, fats, fiber, sugar, sodium
- Categorized food items

### FoodDetectionLog
- Stores uploaded images and detection results
- Confidence scores for detected foods
- User-specific detection history

### DailyNutritionLog
- Daily meal tracking (breakfast, lunch, dinner, snack)
- Custom food entries and quantities
- Notes and additional information

### NutritionGoal
- Daily nutrition targets vs actual intake
- Progress tracking for all macronutrients
- Historical goal data

## AI/ML Integration

### Food Detection
- Uses YOLOv8 for object detection
- Custom food class mapping
- Confidence threshold filtering
- Multi-food detection support

### Nutrition Database
- Local nutrition database with common foods
- Extensible for additional food items
- Default nutrition values for unknown foods

### Dietary Suggestions
- Protein content analysis
- Fiber intake recommendations
- Food variety suggestions

## Development

### Adding New Food Items
1. Access Django admin at `http://localhost:8000/admin/`
2. Navigate to Food Items
3. Add new food with nutrition information

### Customizing Food Detection
1. Train custom YOLOv8 model on food dataset
2. Update `food_detector.py` with new model path
3. Modify class mapping in `_map_class_to_food()`

### Environment Variables
Create a `.env` file for production settings:
```
DEBUG=False
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:password@localhost/dbname
ALLOWED_HOSTS=your-domain.com
```

## Production Deployment

1. **Set DEBUG=False** in settings.py
2. **Configure database** (PostgreSQL recommended)
3. **Set up static files** collection
4. **Configure media file storage**
5. **Set up proper CORS settings**
6. **Use environment variables** for sensitive data

## API Documentation

### Authentication
All protected endpoints require authentication via token:
```
Authorization: Token your-token-here
```

### Example Requests

**Register User:**
```bash
curl -X POST http://localhost:8000/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123",
    "password_confirm": "password123",
    "first_name": "Test",
    "last_name": "User"
  }'
```

**Food Detection:**
```bash
curl -X POST http://localhost:8000/api/detect-food/ \
  -H "Authorization: Token your-token-here" \
  -F "image=@food_image.jpg"
```

**Get Nutrition Summary:**
```bash
curl -X GET "http://localhost:8000/api/nutrition-summary/?date=2024-01-15" \
  -H "Authorization: Token your-token-here"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License. 