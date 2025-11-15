# DataFlow Pro AI 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Next.js](https://img.shields.io/badge/Next.js-13.x-black)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-18.x-blue)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB)](https://www.python.org/)

## 🌟 Overview

DataFlow Pro AI is a comprehensive data analysis and visualization platform built for the GPT-5 Hackathon. This powerful tool helps users transform raw data into actionable insights through automated visualizations, statistical analysis, and interactive dashboards.

![DataFlow Pro AI Dashboard](https://via.placeholder.com/800x400?text=DataFlow+Pro+AI+Dashboard)

## ✨ Features

- **📊 Data Visualization Module** - Generate beautiful, interactive visualizations from CSV and Excel files
- **🤖 AI Chatbot** - Get instant assistance and data insights through natural language queries
- **📈 Data Science Tools** - Apply advanced analytics and statistical methods to your datasets
- **📱 Responsive Dashboard** - Access all features through an intuitive, mobile-friendly interface
- **🎨 Customizable Themes** - Choose between light and dark modes for comfortable viewing

## 🛠️ Technologies

### Frontend
- **Next.js** - React framework for server-side rendering and static generation
- **React** - UI component library
- **Tailwind CSS** - Utility-first CSS framework
- **TypeScript** - Type-safe JavaScript

### Backend
- **FastAPI** - Modern, high-performance web framework for building APIs
- **Python** - Backend programming language
- **Pandas** - Data manipulation and analysis
- **Matplotlib/Seaborn** - Data visualization libraries

## 🚀 Getting Started

### Prerequisites

- Node.js (v16+)
- Python (v3.9+)
- npm or yarn

### Installation

#### Frontend Setup
```bash
# Clone the repository
git clone https://github.com/WasifSohail5/GPT-5-Hackathon.git
cd GPT-5-Hackathon

# Install dependencies
npm install
# or
yarn install

# Create .env.local file with required environment variables
cp .env.example .env.local
```

#### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### Start the Backend Server
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8004 --reload
```

#### Start the Frontend Development Server
```bash
# In a new terminal
npm run dev
# or
yarn dev
```

The application will be available at [http://localhost:3000](http://localhost:3000)

## 📋 Modules

### Report Generator
- Automated data visualization generation
- Statistical analysis summary
- Export reports as PDF/HTML
- Interactive charts and graphs

### Data Science Module
- Correlation analysis
- Descriptive statistics
- Outlier detection
- Time series analysis

### AI Chatbot
- Natural language queries about your data
- Interactive assistance
- Code generation for custom analysis
- Export conversations for documentation

## 📸 Screenshots

### Landing Page
![Landing Page](https://via.placeholder.com/800x400?text=Landing+Page)

### Data Visualization Dashboard
![Visualization Dashboard](https://via.placeholder.com/800x400?text=Data+Visualization+Dashboard)

### AI Chatbot Interface
![AI Chatbot](https://via.placeholder.com/800x400?text=AI+Chatbot+Interface)

## 🔄 Workflow

1. **Upload Data** - Upload CSV or Excel files through the intuitive interface
2. **Generate Visualizations** - Automatically create relevant visualizations based on your data
3. **Explore Insights** - Interact with charts and graphs to discover patterns and trends
4. **Ask Questions** - Use the AI chatbot to ask questions about your data in natural language
5. **Export Results** - Save visualizations and reports for sharing and presentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

- **Wasif Sohail** - [GitHub Profile](https://github.com/WasifSohail5)

## 🙏 Acknowledgements

- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/getting-started.html)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)

---

<div align="center">
  <p>Built with ❤️ for the GPT-5 Hackathon</p>
</div>
