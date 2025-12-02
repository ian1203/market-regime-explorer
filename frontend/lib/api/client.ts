import { API_BASE_URL } from '../config';

export type Metric = {
  model: string;
  accuracy: number;
  f1: number;
  roc_auc: number;
};

export type ClusterStat = {
  cluster_id: number;
  n_days: number;
  mean_daily_return: number;
  prob_up: number;
};

export type SummaryResponse = {
  best_model_name: string;
  metrics: Metric[];
  cluster_stats: ClusterStat[];
  latest_date: string;
  latest_regime: number;
  latest_pred_prob: number;
};

export type TimePoint = {
  date: string;
  spy_price: number;
  true_up: number;
  pred_up_prob: number;
  cluster: number;
};

export type PCAPoint = {
  pc1: number;
  pc2: number;
  cluster_id: number;
  target_up: number;
};

export async function fetchSummary(): Promise<SummaryResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/summary`);
    if (!res.ok) {
      throw new Error(`HTTP error from /summary: ${res.status} ${res.statusText}`);
    }
    return res.json();
  } catch (err) {
    if (err instanceof TypeError && err.message.includes('fetch')) {
      throw new Error(`Network error: Could not connect to ${API_BASE_URL}. Is the backend running?`);
    }
    throw err;
  }
}

export async function fetchTimeseries(): Promise<TimePoint[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/timeseries`);
    if (!res.ok) {
      throw new Error(`HTTP error from /timeseries: ${res.status} ${res.statusText}`);
    }
    const json = await res.json();
    // Backend returns { data: [...] }
    return json.data ?? json;
  } catch (err) {
    if (err instanceof TypeError && err.message.includes('fetch')) {
      throw new Error(`Network error: Could not connect to ${API_BASE_URL}. Is the backend running?`);
    }
    throw err;
  }
}

export async function fetchPCAScatter(): Promise<PCAPoint[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/pca_scatter`);
    if (!res.ok) {
      throw new Error(`HTTP error from /pca_scatter: ${res.status} ${res.statusText}`);
    }
    const json = await res.json();
    // Backend returns { data: [...] }
    return json.data ?? json;
  } catch (err) {
    if (err instanceof TypeError && err.message.includes('fetch')) {
      throw new Error(`Network error: Could not connect to ${API_BASE_URL}. Is the backend running?`);
    }
    throw err;
  }
}

export async function triggerRefresh(): Promise<void> {
  try {
    const res = await fetch(`${API_BASE_URL}/refresh`, {
      method: 'POST',
    });
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`HTTP error from /refresh: ${res.status} ${res.statusText}. ${errorText}`);
    }
  } catch (err) {
    if (err instanceof TypeError && err.message.includes('fetch')) {
      throw new Error(`Network error: Could not connect to ${API_BASE_URL}. Is the backend running?`);
    }
    throw err;
  }
}

export async function askLLM(question: string | null): Promise<string> {
  try {
    const res = await fetch(`${API_BASE_URL}/llm/explain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`LLM API error: ${res.status} – ${text}`);
    }
    const json = await res.json();
    return json.answer as string;
  } catch (err) {
    if (err instanceof TypeError && err.message.includes('fetch')) {
      throw new Error(`Network error: Could not connect to ${API_BASE_URL}. Is the backend running?`);
    }
    throw err;
  }
}

