import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { User, Save, Edit, Target, Activity, Scale, Ruler } from 'lucide-react';
import api from '../api/api';

const Profile = () => {
  const { user, profile, fetchUserProfile } = useAuth();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isEditing, setIsEditing] = useState(false);
  const [profileData, setProfileData] = useState(null);

  const [formData, setFormData] = useState({
    first_name: '',
    last_name: '',
    email: '',
    height: '',
    weight: '',
    age: '',
    gender: 'M',
    activity_level: 'moderately_active',
    target_calories: '',
    target_protein: '',
    target_carbs: '',
    target_fats: ''
  });

  useEffect(() => {
    fetchProfileData();
  }, []);

  useEffect(() => {
    if (user && profileData) {
      setFormData({
        first_name: user.first_name || '',
        last_name: user.last_name || '',
        email: user.email || '',
        height: profileData.height || '',
        weight: profileData.weight || '',
        age: profileData.age || '',
        gender: profileData.gender || 'M',
        activity_level: profileData.activity_level || 'moderately_active',
        target_calories: profileData.target_calories || '',
        target_protein: profileData.target_protein || '',
        target_carbs: profileData.target_carbs || '',
        target_fats: profileData.target_fats || ''
      });
    }
  }, [user, profileData]);

  const fetchProfileData = async () => {
    try {
      setLoading(true);
      const response = await api.get('/profile/');
      setProfileData(response.data);
      setError('');
    } catch (err) {
      console.error('Error fetching profile data:', err);
      setError('Failed to load profile data');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError('');
    setSuccess('');

    try {
      const response = await api.put('/profile/', formData);
      setSuccess('Profile updated successfully!');
      setIsEditing(false);
      await fetchUserProfile(); // Refresh user data
      await fetchProfileData(); // Refresh profile data
    } catch (err) {
      console.error('Error updating profile:', err);
      setError(err.response?.data?.error || 'Failed to update profile');
    } finally {
      setSaving(false);
    }
  };

  const calculateBMI = () => {
    if (profileData?.height && profileData?.weight) {
      const heightInMeters = profileData.height / 100;
      const bmi = profileData.weight / (heightInMeters * heightInMeters);
      return bmi.toFixed(1);
    }
    return null;
  };

  const getBMICategory = (bmi) => {
    if (bmi < 18.5) return { category: 'Underweight', color: 'text-blue-600 bg-blue-100' };
    if (bmi < 25) return { category: 'Normal', color: 'text-green-600 bg-green-100' };
    if (bmi < 30) return { category: 'Overweight', color: 'text-yellow-600 bg-yellow-100' };
    return { category: 'Obese', color: 'text-red-600 bg-red-100' };
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error && !profileData) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-primary-600 to-secondary-600 rounded-lg p-6 text-white">
          <h1 className="text-2xl font-bold mb-2 flex items-center">
            <User className="h-6 w-6 mr-3" />
            Profile Settings
          </h1>
          <p className="text-primary-100">
            Manage your personal information and nutrition goals.
          </p>
        </div>
        <div className="bg-danger-50 border border-danger-200 text-danger-700 px-4 py-3 rounded-lg">
          <p className="font-medium">Error loading profile</p>
          <p className="text-sm mt-1">{error}</p>
          <button
            onClick={fetchProfileData}
            className="btn-primary mt-3"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  const bmi = calculateBMI();
  const bmiCategory = bmi ? getBMICategory(bmi) : null;

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-primary-600 to-secondary-600 rounded-lg p-6 text-white">
        <h1 className="text-2xl font-bold mb-2 flex items-center">
          <User className="h-6 w-6 mr-3" />
          Profile Settings
        </h1>
        <p className="text-primary-100">
          Manage your personal information and nutrition goals.
        </p>
      </div>

      {/* Profile Overview */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900">Profile Overview</h2>
          <button
            onClick={() => setIsEditing(!isEditing)}
            className="btn-secondary flex items-center"
          >
            <Edit className="h-4 w-4 mr-2" />
            {isEditing ? 'Cancel' : 'Edit Profile'}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* User Info */}
          <div className="space-y-4">
            <div className="flex items-center">
              <div className="h-16 w-16 bg-primary-600 rounded-full flex items-center justify-center">
                <span className="text-xl font-bold text-white">
                  {user?.first_name?.[0] || 'U'}
                </span>
              </div>
              <div className="ml-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  {user?.first_name} {user?.last_name}
                </h3>
                <p className="text-gray-600">{user?.email}</p>
              </div>
            </div>

            {/* BMI Info */}
            {bmi && (
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">BMI</p>
                    <p className="text-2xl font-bold text-gray-900">{bmi}</p>
                  </div>
                  {bmiCategory && (
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${bmiCategory.color}`}>
                      {bmiCategory.category}
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Physical Stats */}
          <div className="space-y-4">
            <h4 className="font-medium text-gray-900 flex items-center">
              <Scale className="h-4 w-4 mr-2" />
              Physical Stats
            </h4>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Height:</span>
                <span className="font-medium">{profileData?.height || 'Not set'} cm</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Weight:</span>
                <span className="font-medium">{profileData?.weight || 'Not set'} kg</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Age:</span>
                <span className="font-medium">{profileData?.age || 'Not set'} years</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Gender:</span>
                <span className="font-medium">{profileData?.gender === 'M' ? 'Male' : profileData?.gender === 'F' ? 'Female' : 'Not set'}</span>
              </div>
            </div>
          </div>

          {/* Activity Level */}
          <div className="space-y-4">
            <h4 className="font-medium text-gray-900 flex items-center">
              <Activity className="h-4 w-4 mr-2" />
              Activity Level
            </h4>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600 capitalize">
                {profileData?.activity_level?.replace('_', ' ') || 'Not set'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Nutrition Goals */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-6 flex items-center">
          <Target className="h-5 w-5 mr-2" />
          Nutrition Goals
        </h2>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <p className="text-sm text-red-600 font-medium">Daily Calories</p>
            <p className="text-2xl font-bold text-red-700">{profileData?.target_calories || 'Not set'}</p>
          </div>
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <p className="text-sm text-blue-600 font-medium">Protein (g)</p>
            <p className="text-2xl font-bold text-blue-700">{profileData?.target_protein || 'Not set'}</p>
          </div>
          <div className="text-center p-4 bg-yellow-50 rounded-lg">
            <p className="text-sm text-yellow-600 font-medium">Carbs (g)</p>
            <p className="text-2xl font-bold text-yellow-700">{profileData?.target_carbs || 'Not set'}</p>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <p className="text-sm text-green-600 font-medium">Fats (g)</p>
            <p className="text-2xl font-bold text-green-700">{profileData?.target_fats || 'Not set'}</p>
          </div>
        </div>
      </div>

      {/* Edit Form */}
      {isEditing && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">Edit Profile</h2>

          {error && (
            <div className="bg-danger-50 border border-danger-200 text-danger-700 px-4 py-3 rounded-lg mb-4">
              {error}
            </div>
          )}

          {success && (
            <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg mb-4">
              {success}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="first_name" className="form-label">First Name</label>
                <input
                  id="first_name"
                  name="first_name"
                  type="text"
                  value={formData.first_name}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>
              <div>
                <label htmlFor="last_name" className="form-label">Last Name</label>
                <input
                  id="last_name"
                  name="last_name"
                  type="text"
                  value={formData.last_name}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>
            </div>

            <div>
              <label htmlFor="email" className="form-label">Email</label>
              <input
                id="email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                className="input-field"
                required
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label htmlFor="height" className="form-label">Height (cm)</label>
                <input
                  id="height"
                  name="height"
                  type="number"
                  value={formData.height}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>
              <div>
                <label htmlFor="weight" className="form-label">Weight (kg)</label>
                <input
                  id="weight"
                  name="weight"
                  type="number"
                  value={formData.weight}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>
              <div>
                <label htmlFor="age" className="form-label">Age</label>
                <input
                  id="age"
                  name="age"
                  type="number"
                  value={formData.age}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="gender" className="form-label">Gender</label>
                <select
                  id="gender"
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                  className="input-field"
                >
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </select>
              </div>
              <div>
                <label htmlFor="activity_level" className="form-label">Activity Level</label>
                <select
                  id="activity_level"
                  name="activity_level"
                  value={formData.activity_level}
                  onChange={handleChange}
                  className="input-field"
                >
                  <option value="sedentary">Sedentary</option>
                  <option value="lightly_active">Lightly Active</option>
                  <option value="moderately_active">Moderately Active</option>
                  <option value="very_active">Very Active</option>
                  <option value="extremely_active">Extremely Active</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <label htmlFor="target_calories" className="form-label">Target Calories</label>
                <input
                  id="target_calories"
                  name="target_calories"
                  type="number"
                  value={formData.target_calories}
                  onChange={handleChange}
                  className="input-field"
                />
              </div>
              <div>
                <label htmlFor="target_protein" className="form-label">Target Protein (g)</label>
                <input
                  id="target_protein"
                  name="target_protein"
                  type="number"
                  value={formData.target_protein}
                  onChange={handleChange}
                  className="input-field"
                />
              </div>
              <div>
                <label htmlFor="target_carbs" className="form-label">Target Carbs (g)</label>
                <input
                  id="target_carbs"
                  name="target_carbs"
                  type="number"
                  value={formData.target_carbs}
                  onChange={handleChange}
                  className="input-field"
                />
              </div>
              <div>
                <label htmlFor="target_fats" className="form-label">Target Fats (g)</label>
                <input
                  id="target_fats"
                  name="target_fats"
                  type="number"
                  value={formData.target_fats}
                  onChange={handleChange}
                  className="input-field"
                />
              </div>
            </div>

            <div className="flex justify-end space-x-4">
              <button
                type="button"
                onClick={() => setIsEditing(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={saving}
                className="btn-primary flex items-center"
              >
                {saving ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4 mr-2" />
                    Save Changes
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};

export default Profile;