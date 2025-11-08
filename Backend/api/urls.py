from django.urls import path
from . import views

urlpatterns = [
    # Test endpoint
    path('test/', views.test_api, name='test-api'),
    
    # Welcome endpoint
    path('welcome/', views.welcome_api, name='welcome-api'),

    # Authentication endpoints
    path('auth/register/', views.UserRegistrationView.as_view(), name='user-register'),
    path('auth/login/', views.UserLoginView.as_view(), name='user-login'),
    path('auth/logout/', views.user_logout, name='user-logout'),

    # User profile
    path('profile/', views.UserProfileView.as_view(), name='user-profile'),

    # Food detection
    path('detect-food/', views.FoodDetectionView.as_view(), name='food-detection'),
    path('detection-history/', views.detection_history, name='detection-history'),

    # Food items
    path('food-items/', views.FoodItemListView.as_view(), name='food-items'),

    # Nutrition logging
    path('nutrition-log/', views.DailyNutritionLogView.as_view(), name='nutrition-log'),
    path('nutrition-summary/', views.NutritionSummaryView.as_view(), name='nutrition-summary'),

    # Reports
    path('weekly-report/', views.WeeklyReportView.as_view(), name='weekly-report'),
    
    # Enhanced Nutrition Tracking
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('nutrition-log-enhanced/', views.EnhancedNutritionLogView.as_view(), name='nutrition-log-enhanced'),
    path('nutrition-reports/', views.NutritionReportsView.as_view(), name='nutrition-reports'),
] 