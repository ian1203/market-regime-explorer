import { ClusterStat } from '@/lib/api/client';

interface ClusterStatsTableProps {
  clusterStats: ClusterStat[];
}

export default function ClusterStatsTable({ clusterStats }: ClusterStatsTableProps) {
  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;
  const formatReturn = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${(value * 100).toFixed(3)}%`;
  };

  return (
    <div className="bg-dark-card rounded-lg p-6 shadow-lg border border-dark-border">
      <h2 className="text-xl font-bold mb-4">Market Regime Statistics</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-dark-border">
              <th className="text-left py-3 px-4 text-gray-400 font-semibold">Regime</th>
              <th className="text-right py-3 px-4 text-gray-400 font-semibold">Days</th>
              <th className="text-right py-3 px-4 text-gray-400 font-semibold">Mean Return</th>
              <th className="text-right py-3 px-4 text-gray-400 font-semibold">Prob Up</th>
            </tr>
          </thead>
          <tbody>
            {clusterStats.map((stat) => (
              <tr
                key={stat.cluster_id}
                className="border-b border-dark-border hover:bg-dark-border/50 transition-colors"
              >
                <td className="py-3 px-4">
                  <span className="font-semibold text-white">Regime {stat.cluster_id}</span>
                </td>
                <td className="py-3 px-4 text-right text-gray-300">{stat.n_days.toLocaleString()}</td>
                <td className={`py-3 px-4 text-right ${
                  stat.mean_daily_return >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {formatReturn(stat.mean_daily_return)}
                </td>
                <td className="py-3 px-4 text-right text-gray-300">{formatPercent(stat.prob_up)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

