import { Metric } from '@/lib/api/client';

interface MetricsTableProps {
  metrics: Metric[];
}

export default function MetricsTable({ metrics }: MetricsTableProps) {
  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;

  return (
    <div className="bg-dark-card rounded-lg p-6 shadow-lg border border-dark-border">
      <h2 className="text-xl font-bold mb-4">Model Performance Metrics</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-dark-border">
              <th className="text-left py-3 px-4 text-gray-400 font-semibold">Model</th>
              <th className="text-right py-3 px-4 text-gray-400 font-semibold">Accuracy</th>
              <th className="text-right py-3 px-4 text-gray-400 font-semibold">F1 Score</th>
              <th className="text-right py-3 px-4 text-gray-400 font-semibold">ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric, idx) => (
              <tr
                key={idx}
                className="border-b border-dark-border hover:bg-dark-border/50 transition-colors"
              >
                <td className="py-3 px-4 text-gray-300">{metric.model}</td>
                <td className="py-3 px-4 text-right text-gray-300">{formatPercent(metric.accuracy)}</td>
                <td className="py-3 px-4 text-right text-gray-300">{formatPercent(metric.f1)}</td>
                <td className="py-3 px-4 text-right text-gray-300">{formatPercent(metric.roc_auc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

