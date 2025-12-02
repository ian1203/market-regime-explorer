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

