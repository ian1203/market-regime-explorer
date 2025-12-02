import { useState, useEffect } from 'react';
import Layout from '@/components/Layout';
import SummaryCards from '@/components/SummaryCards';
import MetricsTable from '@/components/MetricsTable';
import ClusterStatsTable from '@/components/ClusterStatsTable';
import TimeSeriesChart from '@/components/TimeSeriesChart';
import PCAScatter from '@/components/PCAScatter';
import Loading from '@/components/Loading';
import {
  fetchSummary,
  fetchTimeseries,
  fetchPCAScatter,
  triggerRefresh,
  askLLM,
  SummaryResponse,
  TimePoint,
  PCAPoint,
} from '@/lib/api/client';

export default function Home() {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [timeseries, setTimeseries] = useState<TimePoint[] | null>(null);
  const [pcaScatter, setPcaScatter] = useState<PCAPoint[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmError, setLlmError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setError(null);

        // Fetch all data in parallel
        const [summaryData, timeseriesData, pcaData] = await Promise.all([
          fetchSummary(),
          fetchTimeseries(),
          fetchPCAScatter(),
        ]);

        setSummary(summaryData);
        setTimeseries(timeseriesData);
        setPcaScatter(pcaData);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to load data';
        setError(errorMessage);
        console.error('Error loading data:', err);
        // Log more details for debugging
        if (err instanceof Error) {
          console.error('Error name:', err.name);
          console.error('Error stack:', err.stack);
        }
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  const handleRefresh = async () => {
    try {
      setRefreshing(true);
      setError(null);
      
      // Trigger backend refresh
      await triggerRefresh();
      
      // Wait a moment for the backend to finish processing
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Reload all data
      const [summaryData, timeseriesData, pcaData] = await Promise.all([
        fetchSummary(),
        fetchTimeseries(),
        fetchPCAScatter(),
      ]);
      
      setSummary(summaryData);
      setTimeseries(timeseriesData);
      setPcaScatter(pcaData);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to refresh data';
      setError(errorMessage);
      console.error('Error refreshing data:', err);
    } finally {
      setRefreshing(false);
    }
  };

  const handleAsk = async () => {
    setLlmLoading(true);
    setLlmError(null);
    try {
      const ans = await askLLM(question.trim() === '' ? null : question.trim());
      setAnswer(ans);
    } catch (err: any) {
      setLlmError(err.message ?? 'Unexpected error');
      console.error('Error asking LLM:', err);
    } finally {
      setLlmLoading(false);
    }
  };

  if (loading) {
    return (
      <Layout onRefresh={handleRefresh} isRefreshing={refreshing}>
        <Loading />
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout onRefresh={handleRefresh} isRefreshing={refreshing}>
        <div className="bg-red-900/20 border border-red-500 rounded-lg p-6">
          <h2 className="text-xl font-bold text-red-400 mb-2">Error Loading Data</h2>
          <p className="text-gray-300 mb-3">{error}</p>
          <div className="text-sm text-gray-400 space-y-1">
            <p>• Make sure the backend API is running on http://localhost:8000</p>
            <p>• Check the browser console for more details</p>
            <p>• Verify CORS is enabled in the backend</p>
          </div>
        </div>
      </Layout>
    );
  }

  if (!summary || !timeseries || !pcaScatter) {
    return (
      <Layout onRefresh={handleRefresh} isRefreshing={refreshing}>
        <div className="text-center text-gray-400">No data available</div>
      </Layout>
    );
  }

  return (
    <Layout onRefresh={handleRefresh} isRefreshing={refreshing}>
      {/* Summary Cards */}
      <SummaryCards data={summary} />

      {/* Time Series Chart */}
      <TimeSeriesChart data={timeseries} />

      {/* AI Insight Panel */}
      <div className="bg-neutral-900/70 border border-neutral-800 rounded-2xl p-6 mb-8">
        <div className="flex items-center gap-2 mb-4">
          <svg
            className="w-5 h-5 text-emerald-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
            />
          </svg>
          <h2 className="text-xl font-bold text-white">AI Insight</h2>
        </div>
        <p className="text-sm text-gray-400 mb-4">
          Ask the AI to explain the current market regime, how it has behaved
          historically, and how that compares to the model&apos;s current next-day
          probability for SPY.
        </p>
        <div className="space-y-4">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about the current market regime, predictions, or leave empty for a default explanation..."
            className="w-full bg-neutral-950/60 border border-neutral-800 rounded-xl px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/60 resize-none"
            rows={3}
          />
          <button
            onClick={handleAsk}
            disabled={llmLoading}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors text-sm font-semibold"
          >
            {llmLoading ? 'Asking AI...' : 'Ask AI'}
          </button>
          {llmError && (
            <div className="bg-red-900/20 border border-red-500 rounded-lg p-3 text-sm text-red-400">
              {llmError}
            </div>
          )}
          {answer && (
            <div className="bg-neutral-950/60 border border-neutral-800 rounded-xl p-4 max-h-56 overflow-y-auto">
              <p className="text-sm text-neutral-100 whitespace-pre-wrap">{answer}</p>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Section: PCA Scatter and Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: PCA Scatter */}
        <div>
          <PCAScatter data={pcaScatter} />
        </div>

        {/* Right: Tables */}
        <div className="space-y-6">
          <ClusterStatsTable clusterStats={summary.cluster_stats} />
          <MetricsTable metrics={summary.metrics} />
        </div>
      </div>
    </Layout>
  );
}

