import { PCAPoint } from '@/lib/api/client';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface PCAScatterProps {
  data: PCAPoint[];
}

export default function PCAScatter({ data }: PCAScatterProps) {
  // Prepare data for Recharts (expects array of objects with x, y, and optional z)
  const chartData = data.map((point) => ({
    x: point.pc1,
    y: point.pc2,
    cluster: point.cluster_id,
    targetUp: point.target_up,
  }));

  // Separate data by cluster for coloring
  const clusters = Array.from(new Set(data.map((d) => d.cluster_id))).sort();
  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const point = payload[0].payload;
      return (
        <div className="bg-dark-card border border-dark-border rounded-lg p-3 shadow-lg">
          <p className="text-white">
            <span className="text-gray-400">PC1: </span>
            <span className="font-semibold">{point.x.toFixed(3)}</span>
          </p>
          <p className="text-white">
            <span className="text-gray-400">PC2: </span>
            <span className="font-semibold">{point.y.toFixed(3)}</span>
          </p>
          <p className="text-white">
            <span className="text-gray-400">Regime: </span>
            <span className="font-semibold">{point.cluster}</span>
          </p>
          <p className="text-white">
            <span className="text-gray-400">Target Up: </span>
            <span className="font-semibold">{point.targetUp === 1 ? 'Yes' : 'No'}</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-dark-card rounded-lg p-6 shadow-lg border border-dark-border">
      <h2 className="text-xl font-bold mb-4">PCA Scatter Plot (Training Data)</h2>
      <p className="text-sm text-gray-400 mb-4">
        First two principal components colored by market regime
      </p>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
          <XAxis
            type="number"
            dataKey="x"
            name="PC1"
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af' }}
            label={{ value: 'PC1', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="PC2"
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af' }}
            label={{ value: 'PC2', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
          <Legend
            wrapperStyle={{ color: '#9ca3af' }}
          />
          {clusters.map((clusterId, idx) => {
            const clusterData = chartData.filter((d) => d.cluster === clusterId);
            return (
              <Scatter
                key={clusterId}
                name={`Regime ${clusterId}`}
                data={clusterData}
                fill={colors[idx % colors.length]}
                opacity={0.6}
              />
            );
          })}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

