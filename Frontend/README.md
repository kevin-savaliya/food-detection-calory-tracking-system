# AI Calorie Tracking System - Frontend

A beautiful and modern React frontend for the AI Calorie Tracking System with comprehensive nutrition tracking, food detection, and analytics.

## 🚀 Features

### 📊 Dashboard
- **Real-time Progress Tracking**: Visual progress bars for calories, protein, carbs, and fats
- **Weekly Trends**: Interactive charts showing nutrition trends over time
- **Achievement Insights**: Streak tracking and goal achievement rates
- **Quick Stats**: Remaining nutrients and daily targets
- **Recent Meals**: Latest food entries with detailed nutrition breakdown

### 📸 Food Detection
- **AI-Powered Recognition**: Upload food images for automatic detection
- **Nutrition Calculation**: Automatic calorie and macro calculation
- **Confidence Scoring**: Visual confidence indicators for each detection
- **Auto-Logging**: Detected foods automatically added to daily log
- **Image Preview**: Real-time preview of uploaded images

### 📝 Nutrition Log
- **Meal Organization**: Separate sections for breakfast, lunch, dinner, and snacks
- **Manual Entry**: Add custom food entries with full nutrition data
- **Date Navigation**: Browse nutrition logs for any date
- **Edit/Delete**: Modify or remove existing entries
- **Progress Tracking**: Real-time progress toward daily goals

### 📈 Reports & Analytics
- **Multiple Timeframes**: Weekly, monthly, and yearly reports
- **Visual Charts**: Interactive line charts and pie charts
- **Top Foods**: Most consumed foods with frequency and calories
- **Meal Breakdown**: Distribution of calories across meal types
- **Achievement Rates**: Goal achievement percentages for all nutrients

## 🎨 UI/UX Features

- **Modern Material Design**: Clean, intuitive interface using Material-UI
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **Beautiful Gradients**: Eye-catching gradient cards and backgrounds
- **Interactive Charts**: Dynamic charts with hover effects and tooltips
- **Real-time Updates**: Instant feedback and data updates
- **Accessibility**: Full keyboard navigation and screen reader support

## 🛠️ Installation

1. **Navigate to Frontend Directory**:
   ```bash
   cd Frontend
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   ```

3. **Start Development Server**:
   ```bash
   npm start
   ```

4. **Open in Browser**:
   ```
   http://localhost:3000
   ```

## 📦 Dependencies

### Core Libraries
- **React 18**: Modern React with hooks and functional components
- **Material-UI 5**: Complete UI component library
- **Axios**: HTTP client for API communication
- **Recharts**: Beautiful charts and data visualization

### Date Handling
- **@mui/x-date-pickers**: Advanced date picker components
- **date-fns**: Modern date utility library

### Icons & Styling
- **@mui/icons-material**: Comprehensive icon set
- **@emotion**: CSS-in-JS styling solution

## 🔧 Configuration

### API Endpoints
The frontend is configured to connect to the Django backend at `http://localhost:8000`. Update the API base URL in components if needed:

```javascript
const API_BASE_URL = 'http://localhost:8000/api';
```

### Authentication
- Uses token-based authentication
- Stores JWT tokens in localStorage
- Automatic token refresh and error handling

## 📱 Component Structure

```
src/
├── components/
│   ├── Dashboard.js          # Main dashboard with progress tracking
│   ├── FoodDetection.js      # AI food detection interface
│   ├── NutritionLog.js       # Daily nutrition logging
│   └── Reports.js           # Analytics and reports
├── App.js                   # Main application component
└── package.json            # Dependencies and scripts
```

## 🎯 Key Components

### Dashboard Component
- **Progress Cards**: Visual representation of daily nutrition goals
- **Weekly Chart**: Line chart showing nutrition trends
- **Recent Meals**: List of latest food entries
- **Insights Panel**: Streak tracking and remaining nutrients

### Food Detection Component
- **Image Upload**: Drag-and-drop or click to upload
- **Detection Results**: Detailed nutrition breakdown
- **Confidence Indicators**: Visual confidence scores
- **Auto-Logging**: Automatic addition to nutrition log

### Nutrition Log Component
- **Meal Sections**: Organized by meal type (breakfast, lunch, dinner, snack)
- **Manual Entry Dialog**: Add custom food entries
- **Date Picker**: Navigate between different dates
- **Edit/Delete Actions**: Modify existing entries

### Reports Component
- **Time Period Selection**: Weekly, monthly, yearly views
- **Interactive Charts**: Line charts for trends, pie charts for distribution
- **Data Tables**: Top foods and meal breakdown tables
- **Achievement Metrics**: Goal achievement rates and statistics

## 🎨 Design System

### Color Palette
- **Primary**: Blue (#1976d2) - Main actions and highlights
- **Secondary**: Pink (#dc004e) - Secondary actions and accents
- **Success**: Green (#4caf50) - Positive states and achievements
- **Warning**: Orange (#ff9800) - Caution and attention
- **Error**: Red (#f44336) - Errors and warnings

### Typography
- **Headers**: Bold, modern fonts for clear hierarchy
- **Body Text**: Readable fonts for content
- **Captions**: Smaller text for secondary information

### Spacing & Layout
- **Consistent Spacing**: 8px grid system
- **Card-based Layout**: Clean card containers for content
- **Responsive Grid**: Flexible layout that adapts to screen size

## 🔄 Data Flow

1. **Authentication**: User logs in, token stored in localStorage
2. **API Calls**: All components make authenticated API calls to Django backend
3. **Real-time Updates**: Data refreshes automatically after actions
4. **Error Handling**: Graceful error handling with user-friendly messages
5. **Loading States**: Loading indicators for better UX

## 🚀 Getting Started

1. **Make sure Django backend is running** on port 8000
2. **Install frontend dependencies**: `npm install`
3. **Start the frontend**: `npm start`
4. **Open browser**: Navigate to `http://localhost:3000`
5. **Login or register** to access the nutrition tracking features

## 📊 Features Overview

| Feature | Description | Status |
|---------|-------------|--------|
| Dashboard | Real-time nutrition progress | ✅ Complete |
| Food Detection | AI-powered food recognition | ✅ Complete |
| Nutrition Log | Daily meal tracking | ✅ Complete |
| Reports | Analytics and insights | ✅ Complete |
| Manual Entry | Custom food logging | ✅ Complete |
| Progress Tracking | Goal achievement monitoring | ✅ Complete |
| Responsive Design | Mobile-friendly interface | ✅ Complete |
| Real-time Updates | Live data synchronization | ✅ Complete |

## 🎉 Ready to Use!

The frontend is fully functional and ready to use with the Django backend. All components are beautifully designed with modern UI patterns and provide a comprehensive nutrition tracking experience.

**Next Steps:**
1. Start the Django backend server
2. Install and start the frontend
3. Register a new user or login
4. Upload food images and start tracking your nutrition!