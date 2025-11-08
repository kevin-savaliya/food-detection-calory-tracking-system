import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { 
  Calendar, 
  TrendingUp, 
  Target, 
  Zap, 
  Activity,
  Clock,
  Award,
  BarChart3
} from 'lucide-react';
import api from '../api/api';

const Dashboard = () => {
  const { user, profile } = useAuth();
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchDashboardData();
  }, []);

  // Listen for food detection completion to refresh data
  useEffect(() => {
    const handleFoodDetectionComplete = () => {
      fetchDashboardData();
    };

    window.addEventListener('foodDetectionComplete', handleFoodDetectionComplete);
    window.addEventListener('focus', handleFoodDetectionComplete);

    return () => {
      window.removeEventListener('foodDetectionComplete', handleFoodDetectionComplete);
      window.removeEventListener('focus', handleFoodDetectionComplete);
    };
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await api.get('/dashboard/');
      setDashboardData(response.data.dashboard_data);
      setError('');
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const getProgressColor = (percentage) => {
    if (percentage >= 100) return 'text-green-600 bg-green-100';
    if (percentage >= 75) return 'text-blue-600 bg-blue-100';
    if (percentage >= 50) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getProgressBarColor = (percentage) => {
    if (percentage >= 100) return 'bg-green-500';
    if (percentage >= 75) return 'bg-blue-500';
    if (percentage >= 50) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-danger-50 border border-danger-200 text-danger-700 px-4 py-3 rounded-lg">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-primary-600 to-secondary-600 rounded-lg p-6 text-white">
        <h1 className="text-2xl font-bold mb-2">
          Welcome back, {user?.first_name || 'User'}! 👋
        </h1>
        <p className="text-primary-100">
          Track your nutrition and achieve your health goals today.
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center">
            <div className="p-2 bg-red-100 rounded-lg">
              <Zap className="h-6 w-6 text-red-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Calories</p>
              <p className="text-2xl font-bold text-gray-900">
                {dashboardData?.today?.nutrition?.calories || 0}
              </p>
              <p className="text-xs text-gray-500">
                / {dashboardData?.today?.targets?.calories || 0} goal
              </p>
            </div>
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${getProgressBarColor(dashboardData?.today?.progress?.calories || 0)}`}
                style={{ width: `${Math.min(dashboardData?.today?.progress?.calories || 0, 100)}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              {dashboardData?.today?.progress?.calories || 0}% of goal
            </p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Target className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Protein</p>
              <p className="text-2xl font-bold text-gray-900">
                {dashboardData?.today?.nutrition?.protein || 0}g
              </p>
              <p className="text-xs text-gray-500">
                / {dashboardData?.today?.targets?.protein || 0}g goal
              </p>
            </div>
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${getProgressBarColor(dashboardData?.today?.progress?.protein || 0)}`}
                style={{ width: `${Math.min(dashboardData?.today?.progress?.protein || 0, 100)}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              {dashboardData?.today?.progress?.protein || 0}% of goal
            </p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center">
            <div className="p-2 bg-yellow-100 rounded-lg">
              <Activity className="h-6 w-6 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Carbs</p>
              <p className="text-2xl font-bold text-gray-900">
                {dashboardData?.today?.nutrition?.carbs || 0}g
              </p>
              <p className="text-xs text-gray-500">
                / {dashboardData?.today?.targets?.carbs || 0}g goal
              </p>
            </div>
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${getProgressBarColor(dashboardData?.today?.progress?.carbs || 0)}`}
                style={{ width: `${Math.min(dashboardData?.today?.progress?.carbs || 0, 100)}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              {dashboardData?.today?.progress?.carbs || 0}% of goal
            </p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center">
            <div className="p-2 bg-green-100 rounded-lg">
              <TrendingUp className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Fats</p>
              <p className="text-2xl font-bold text-gray-900">
                {dashboardData?.today?.nutrition?.fats || 0}g
              </p>
              <p className="text-xs text-gray-500">
                / {dashboardData?.today?.targets?.fats || 0}g goal
              </p>
            </div>
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${getProgressBarColor(dashboardData?.today?.progress?.fats || 0)}`}
                style={{ width: `${Math.min(dashboardData?.today?.progress?.fats || 0, 100)}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              {dashboardData?.today?.progress?.fats || 0}% of goal
            </p>
          </div>
        </div>
      </div>

      {/* Recent Meals */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center">
            <Clock className="h-5 w-5 mr-2 text-gray-600" />
            Recent Meals
          </h2>
          <span className="text-sm text-gray-500">
            {dashboardData?.today?.date || new Date().toISOString().split('T')[0]}
          </span>
        </div>
        
        {dashboardData?.recent_meals?.length > 0 ? (
          <div className="space-y-3">
            {dashboardData.recent_meals.map((meal, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  <div className="p-2 bg-primary-100 rounded-lg mr-3">
                    <Award className="h-4 w-4 text-primary-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{meal.food_name}</p>
                    <p className="text-sm text-gray-600 capitalize">{meal.meal_type}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-semibold text-gray-900">{meal.calories} cal</p>
                  <p className="text-xs text-gray-500">
                    {meal.protein}g P • {meal.carbs}g C • {meal.fats}g F
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No meals logged today yet.</p>
            <p className="text-sm text-gray-400 mt-1">
              Start by detecting some food or adding a meal manually.
            </p>
          </div>
        )}
      </div>

      {/* Weekly Progress */}
      {dashboardData?.weekly_progress && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Calendar className="h-5 w-5 mr-2 text-gray-600" />
            Weekly Progress
          </h2>
          <div className="grid grid-cols-7 gap-2">
            {dashboardData.weekly_progress.map((day, index) => (
              <div key={index} className="text-center">
                <div className="text-xs text-gray-600 mb-1">
                  {new Date(day.date).toLocaleDateString('en', { weekday: 'short' })}
                </div>
                <div className="h-16 bg-gray-100 rounded-lg flex items-end justify-center p-1">
                  {(() => {
                    const target = dashboardData?.today?.targets?.calories || 1;
                    const rawPct = (day.calories / target) * 100;
                    const pct = Math.max(0, Math.min(rawPct, 100));
                    return (
                      <div
                        className="bg-green-500 rounded w-full"
                        style={{ height: `${pct}%` }}
                      ></div>
                    );
                  })()}
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {day.calories || 0}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Insights */}
      {dashboardData?.insights && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Today's Insights</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <h3 className="font-medium text-blue-900 mb-2">Tracking Streak</h3>
              <p className="text-2xl font-bold text-blue-600">
                {dashboardData.insights.tracking_streak || 0} days
              </p>
              <p className="text-sm text-blue-700">Keep it up! 🎉</p>
            </div>
            <div className="p-4 bg-green-50 rounded-lg">
              <h3 className="font-medium text-green-900 mb-2">Calories Remaining</h3>
              <p className="text-2xl font-bold text-green-600">
                {dashboardData.insights.calories_remaining || 0}
              </p>
              <p className="text-sm text-green-700">calories left today</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
