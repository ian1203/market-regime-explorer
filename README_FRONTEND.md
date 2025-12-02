# Market Regime Explorer - Frontend

A modern Next.js dashboard for visualizing market regime detection and prediction results.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Make sure the backend API is running on `http://localhost:8000`:
```bash
# In a separate terminal, from the project root:
uvicorn backend.api:app --reload
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
├── pages/
│   ├── _app.tsx          # Next.js app wrapper
│   └── index.tsx         # Main dashboard page
├── components/
│   ├── Layout.tsx        # Top-level layout with header
│   ├── SummaryCards.tsx  # Summary cards (best model, regime, prediction)
│   ├── MetricsTable.tsx  # Model performance metrics table
│   ├── ClusterStatsTable.tsx  # Market regime statistics table
│   ├── TimeSeriesChart.tsx    # SPY price & prediction chart
│   ├── PCAScatter.tsx    # PCA scatter plot
│   └── Loading.tsx       # Loading component
├── lib/
│   └── api/
│       └── client.ts     # API client with TypeScript types
└── styles/
    └── globals.css       # Global styles and Tailwind setup
```

## Features

- **Summary Dashboard**: View best model, current regime, and latest prediction
- **Time Series Visualization**: SPY price and predicted probability over test period
- **PCA Scatter Plot**: Visualize market regimes in PCA space
- **Performance Metrics**: Compare model accuracy, F1 score, and ROC-AUC
- **Regime Statistics**: View statistics for each detected market regime

## Tech Stack

- **Next.js 14** (Pages Router)
- **TypeScript**
- **TailwindCSS** (Dark theme)
- **Recharts** (Data visualization)

## Environment Variables

You can customize the API URL by setting `NEXT_PUBLIC_API_URL` in a `.env.local` file:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Build for Production

```bash
npm run build
npm start
```

