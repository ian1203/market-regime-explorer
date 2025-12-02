import { SummaryResponse } from '@/lib/api/client';

interface SummaryCardsProps {
  data: SummaryResponse;
}

export default function SummaryCards({ data }: SummaryCardsProps) {
  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;
  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      {/* Best Model Card */}
      <div className="bg-dark-card rounded-lg p-6 shadow-lg border border-dark-border">
        <h3 className="text-sm font-semibold text-gray-400 uppercase mb-2">Best Model</h3>
        <p className="text-xl font-bold text-white">{data.best_model_name}</p>
        <p className="text-sm text-gray-400 mt-2">Selected based on ROC-AUC and F1 score</p>
      </div>

      {/* Latest Regime Card */}
      <div className="bg-dark-card rounded-lg p-6 shadow-lg border border-dark-border">
        <h3 className="text-sm font-semibold text-gray-400 uppercase mb-2">Current Regime</h3>
        <p className="text-3xl font-bold text-white mb-1">Regime {data.latest_regime}</p>
        <p className="text-sm text-gray-400">As of {formatDate(data.latest_date)}</p>
      </div>

      {/* Latest Prediction Card */}
      <div className="bg-dark-card rounded-lg p-6 shadow-lg border border-dark-border">
        <h3 className="text-sm font-semibold text-gray-400 uppercase mb-2">Next Day Prediction</h3>
        <p className="text-3xl font-bold text-white mb-1">{formatPercent(data.latest_pred_prob)}</p>
        <p className="text-sm text-gray-400">Probability of SPY closing up</p>
      </div>
    </div>
  );
}

