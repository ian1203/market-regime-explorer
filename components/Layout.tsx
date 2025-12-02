import { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export default function Layout({ children, onRefresh, isRefreshing }: LayoutProps) {
  return (
    <div className="min-h-screen bg-dark-bg text-gray-100">
      <header className="border-b border-dark-border bg-dark-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold">Market Regime Explorer</h1>
              <p className="text-gray-400 mt-1">AI-powered market regime detection and prediction</p>
            </div>
            {onRefresh && (
              <button
                onClick={onRefresh}
                disabled={isRefreshing}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors text-sm font-semibold"
              >
                {isRefreshing ? 'Refreshing...' : 'Refresh Data'}
              </button>
            )}
          </div>
        </div>
      </header>
      <main className="container mx-auto px-4 py-8">
        {children}
      </main>
    </div>
  );
}

