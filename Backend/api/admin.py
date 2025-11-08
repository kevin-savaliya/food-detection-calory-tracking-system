from django.contrib import admin
from .models import UserProfile, FoodItem, FoodDetectionLog, DailyNutritionLog, NutritionGoal

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'age', 'gender', 'height', 'weight', 'activity_level', 'target_calories']
    list_filter = ['gender', 'activity_level']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(FoodItem)
class FoodItemAdmin(admin.ModelAdmin):
    list_display = ['name', 'calories', 'protein', 'carbs', 'fats', 'category']
    list_filter = ['category']
    search_fields = ['name']
    readonly_fields = ['created_at']

@admin.register(FoodDetectionLog)
class FoodDetectionLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'processed_at', 'detected_foods_count']
    list_filter = ['processed_at']
    search_fields = ['user__username']
    readonly_fields = ['processed_at']
    
    def detected_foods_count(self, obj):
        return len(obj.detected_foods) if obj.detected_foods else 0
    detected_foods_count.short_description = 'Detected Foods'

@admin.register(DailyNutritionLog)
class DailyNutritionLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'date', 'meal_type', 'food_item', 'custom_food_name', 'calories']
    list_filter = ['date', 'meal_type']
    search_fields = ['user__username', 'custom_food_name']
    readonly_fields = ['created_at']

@admin.register(NutritionGoal)
class NutritionGoalAdmin(admin.ModelAdmin):
    list_display = ['user', 'date', 'target_calories', 'actual_calories', 'calories_progress']
    list_filter = ['date']
    search_fields = ['user__username']
    readonly_fields = ['created_at', 'updated_at']
    
    def calories_progress(self, obj):
        if obj.target_calories > 0:
            return f"{(obj.actual_calories / obj.target_calories) * 100:.1f}%"
        return "0%"
    calories_progress.short_description = 'Calories Progress'
