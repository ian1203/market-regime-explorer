import { TimePoint } from '@/lib/api/client';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Area,
  AreaChart,
} from 'recharts';

interface TimeSeriesChartProps {
  data: TimePoint[];
}

export default function TimeSeriesChart({ data }: TimeSeriesChartProps) {
  // Format data for Recharts
  const chartData = data.map((point) => ({
    date: new Date(point.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    fullDate: point.date,
    spyPrice: point.spy_price,
    predProb: point.pred_up_prob * 100, // Convert to percentage
    trueUp: point.true_up,
    cluster: point.cluster,
  }));

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-dark-card border border-dark-border rounded-lg p-3 shadow-lg">
          <p className="text-gray-400 text-sm mb-2">{data.fullDate}</p>
          <p className="text-white">
            <span className="text-gray-400">SPY Price: </span>
            <span className="font-semibold">${data.spyPrice.toFixed(2)}</span>
          </p>
          <p className="text-white">
            <span className="text-gray-400">Predicted Up: </span>
            <span className="font-semibold">{data.predProb.toFixed(1)}%</span>
          </p>
          <p className="text-white">
            <span className="text-gray-400">Regime: </span>
            <span className="font-semibold">{data.cluster}</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-dark-card rounded-lg p-6 shadow-lg border border-dark-border mb-8">
      <h2 className="text-xl font-bold mb-4">SPY Price & Prediction Probability (Test Period)</h2>
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
          <XAxis
            dataKey="date"
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af' }}
            angle={-45}
            textAnchor="end"
            height={80}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="left"
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af' }}
            label={{ value: 'SPY Price ($)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af' }}
            domain={[0, 100]}
            label={{ value: 'Predicted Up (%)', angle: 90, position: 'insideRight', fill: '#9ca3af' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ color: '#9ca3af' }}
            iconType="line"
          />
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="spyPrice"
            fill="#3b82f6"
            fillOpacity={0.2}
            stroke="#3b82f6"
            strokeWidth={2}
            name="SPY Price"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="predProb"
            stroke="#10b981"
            strokeWidth={2}
            dot={false}
            name="Predicted Up (%)"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

