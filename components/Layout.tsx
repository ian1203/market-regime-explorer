import { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export default function Layout({ children, onRefresh, isRefreshing }: LayoutProps) {
  return (
    <div className="min-h-screen bg-dark-bg text-gray-100">
      <header className="w-full border-b border-neutral-800 bg-gradient-to-r from-slate-950 via-slate-900 to-black">
        <div className="mx-auto max-w-6xl px-4 py-5 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm shadow-lg">
              MR
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-semibold text-white">Market Regime Explorer</h1>
              <p className="text-sm text-gray-400 mt-1">
                Explore market regimes and next-day SPY predictions with PCA, clustering and ML models.
              </p>
            </div>
          </div>
          {onRefresh && (
            <button
              onClick={onRefresh}
              disabled={isRefreshing}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors text-sm font-semibold shadow-md"
            >
              {isRefreshing ? 'Refreshing...' : 'Refresh Data'}
            </button>
          )}
        </div>
      </header>
      <main className="container mx-auto px-4 py-8">
        {children}
      </main>
    </div>
  );
}

