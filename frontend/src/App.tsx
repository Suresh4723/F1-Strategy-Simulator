// src/App.tsx
import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend
} from 'recharts';
import './index.css';

const API = 'http://localhost:8000/api';

// ─── Types ────────────────────────────────────────────────

interface LapResult {
  lap: number;
  actual: number;
  simulated: number;
  diff: number;
  cumulative_diff: number;
  compound: string;
  event: string;
  stint_lap: number;
}

interface StintSummary {
  stint: number;
  compound: string;
  start_lap: number;
  end_lap: number;
  laps: number;
  avg_actual: number;
  avg_sim: number;
  avg_diff: number;
  stint_actual: number;
  stint_sim: number;
  stint_diff: number;
}

interface PitSummary {
  lap: number;
  actual: number;
  simulated: number;
  diff: number;
  compound: string;
  event: string;
}

interface SimResult {
  actual_strategy: { pit_laps: number[]; compounds: string[] };
  sim_strategy: { pit_laps: number[]; compounds: string[] };
  laps: LapResult[];
  stint_summary: StintSummary[];
  pit_summary: PitSummary[];
  actual_total: number;
  sim_total: number;
  gain_loss: number;
  lap_mae: number;
  lap_rmse: number;
  mode: string;
}

interface RaceInfo {
  year: number;
  race: string;
  drivers: string[];
  compounds: string[];
  total_laps: number;
  has_sc_vsc: boolean;
  sc_vsc_laps: number[];
}

interface Stint {
  pit_after: number | null;
  compound: string;
}

// ─── Constants ────────────────────────────────────────────

const COMPOUND_COLORS: Record<string, string> = {
  SOFT: '#ff3333',
  MEDIUM: '#ffcc00',
  HARD: '#cccccc',
  INTERMEDIATE: '#39b54a',
  WET: '#0072c6',
};

const COMPOUND_BG: Record<string, string> = {
  SOFT: 'rgba(255,51,51,0.15)',
  MEDIUM: 'rgba(255,204,0,0.15)',
  HARD: 'rgba(204,204,204,0.15)',
  INTERMEDIATE: 'rgba(57,181,74,0.15)',
  WET: 'rgba(0,114,198,0.15)',
};

// ─── Utility Components ──────────────────────────────────

function CompoundBadge({ compound }: { compound: string }) {
  const color = COMPOUND_COLORS[compound] || '#888';
  const bg = COMPOUND_BG[compound] || 'rgba(136,136,136,0.15)';
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 10px',
        borderRadius: 20,
        fontSize: '0.75rem',
        fontWeight: 600,
        letterSpacing: '0.5px',
        background: bg,
        color: color,
        border: `1px solid ${color}40`,
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: color,
          flexShrink: 0,
        }}
      />
      {compound}
    </span>
  );
}

function formatTime(seconds: number): string {
  if (seconds <= 0) return '0.000s';
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(3);
  return mins > 0 ? `${mins}:${secs.padStart(6, '0')}` : `${secs}s`;
}

function formatDelta(seconds: number): string {
  const sign = seconds > 0 ? '+' : '';
  return `${sign}${seconds.toFixed(3)}s`;
}

// ─── Main App ─────────────────────────────────────────────

export default function App() {
  const [races, setRaces] = useState<any[]>([]);
  const [selectedYear, setSelectedYear] = useState<number | ''>('');
  const [selectedRace, setSelectedRace] = useState('');
  const [selectedDriver, setSelectedDriver] = useState('');
  const [raceInfo, setRaceInfo] = useState<RaceInfo | null>(null);
  const [stints, setStints] = useState<Stint[]>([
    { pit_after: 20, compound: 'SOFT' },
    { pit_after: null, compound: 'HARD' },
  ]);
  const [result, setResult] = useState<SimResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingRaces, setLoadingRaces] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'chart' | 'table' | 'stints'>('chart');

  // ── Load races ──
  useEffect(() => {
    fetch(`${API}/races`)
      .then((r) => r.json())
      .then((d) => {
        setRaces(d);
        setLoadingRaces(false);
      })
      .catch(() => {
        setError('Cannot connect to backend');
        setLoadingRaces(false);
      });
  }, []);

  const years = Array.from(new Set(races.map((r: any) => r.Year))).sort(
    (a: number, b: number) => b - a
  );
  const raceNames = races
    .filter((r: any) => r.Year === selectedYear)
    .map((r: any) => r.Race);

  // ── Load race info ──
  const loadRaceInfo = useCallback(async () => {
    if (!selectedYear || !selectedRace) return;
    try {
      const res = await fetch(
        `${API}/race-info?year=${selectedYear}&race=${encodeURIComponent(selectedRace)}`
      );
      const info = await res.json();
      setRaceInfo(info);
      if (info.drivers?.length > 0) setSelectedDriver(info.drivers[0]);
      setError(null);
    } catch {
      setError('Failed to load race info');
    }
  }, [selectedYear, selectedRace]);

  useEffect(() => {
    loadRaceInfo();
  }, [loadRaceInfo]);

  // ── Load actual strategy ──
  const loadActualStrategy = async () => {
    if (!selectedYear || !selectedRace || !selectedDriver) return;
    try {
      const res = await fetch(
        `${API}/driver-strategy?year=${selectedYear}&race=${encodeURIComponent(selectedRace)}&driver=${selectedDriver}`
      );
      const data = await res.json();
      const newStints: Stint[] = data.compounds.map((c: string, i: number) => ({
        compound: c,
        pit_after: i < data.pit_laps.length ? data.pit_laps[i] : null,
      }));
      setStints(newStints);
    } catch {
      setError('Failed to load actual strategy');
    }
  };

  // ── Stint management ──
  const addStint = () => {
    const last = stints[stints.length - 1];
    const prevPit =
      last.pit_after ||
      (raceInfo?.total_laps ? Math.floor(raceInfo.total_laps * 0.7) : 40);
    setStints([
      ...stints.slice(0, -1),
      { ...last, pit_after: prevPit },
      { pit_after: null, compound: 'HARD' },
    ]);
  };

  const removeStint = (idx: number) => {
    if (stints.length <= 1) return;
    const ns = stints.filter((_, i) => i !== idx);
    ns[ns.length - 1].pit_after = null;
    setStints(ns);
  };

  const updateStint = (idx: number, field: keyof Stint, value: any) => {
    const ns = [...stints];
    (ns[idx] as any)[field] = value;
    setStints(ns);
  };

  // ── Run simulation ──
  const runSimulation = async () => {
    if (!selectedYear || !selectedRace || !selectedDriver) return;
    setLoading(true);
    setResult(null);
    setError(null);

    const pitLaps = stints
      .filter((s) => s.pit_after !== null)
      .map((s) => s.pit_after as number);
    const compounds = stints.map((s) => s.compound);

    try {
      const res = await fetch(`${API}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          year: selectedYear,
          race: selectedRace,
          driver: selectedDriver,
          pit_laps: pitLaps,
          compounds,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setResult(data);
        setActiveTab('chart');
      } else {
        setError(data.detail || 'Simulation failed');
      }
    } catch {
      setError('Connection error — is the backend running?');
    }
    setLoading(false);
  };

  // ── Chart data ──
  const chartData =
    result?.laps
      .filter((l) => l.event !== 'pit' && l.event !== 'pit+SC')
      .map((l) => ({
        lap: l.lap,
        actual: l.actual,
        simulated: l.simulated,
        diff: l.diff,
      })) || [];

  const cumulativeData =
    result?.laps.map((l) => ({
      lap: l.lap,
      cumDiff: l.cumulative_diff,
      event: l.event,
    })) || [];

  // ─── RENDER ─────────────────────────────────────────────

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-content">
          <div className="header-logo">
            <span className="header-icon">🏎️</span>
            <div>
              <h1>F1 Strategy Lab</h1>
              <p>Race Strategy Simulator</p>
            </div>
          </div>
          <div className="header-badge">
            <span>2022–2025</span>
          </div>
        </div>
      </header>

      {/* ── Error banner ── */}
      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* ── Race Selection ── */}
      <div className="card">
        <div className="card-title">
          <span className="card-icon">📍</span>
          Race Selection
        </div>
        <div className="form-grid">
          <div className="form-group">
            <label>Season</label>
            <select
              value={selectedYear}
              onChange={(e) => {
                setSelectedYear(Number(e.target.value));
                setSelectedRace('');
                setRaceInfo(null);
                setResult(null);
              }}
            >
              <option value="">Select Year</option>
              {years.map((y: number) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Grand Prix</label>
            <select
              value={selectedRace}
              onChange={(e) => {
                setSelectedRace(e.target.value);
                setResult(null);
              }}
            >
              <option value="">Select Race</option>
              {raceNames.map((r: string) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Driver</label>
            <select
              value={selectedDriver}
              onChange={(e) => {
                setSelectedDriver(e.target.value);
                setResult(null);
              }}
            >
              <option value="">Select Driver</option>
              {raceInfo?.drivers.map((d: string) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>
        </div>

        {raceInfo && (
          <div className="race-meta">
            <span>📏 {raceInfo.total_laps} laps</span>
            <span>
              🏁 {raceInfo.compounds?.length || 0} compounds available
            </span>
            {raceInfo.has_sc_vsc && (
              <span className="sc-badge">⚠️ SC/VSC in this race</span>
            )}
          </div>
        )}
      </div>

      {/* ── Strategy Builder ── */}
      {raceInfo && (
        <div className="card">
          <div className="card-title">
            <span className="card-icon">🔧</span>
            Strategy Builder
          </div>

          <div className="strategy-timeline">
            {stints.map((stint, i) => (
              <div className="stint-block" key={i}>
                <div
                  className="stint-header"
                  style={{
                    borderLeft: `3px solid ${COMPOUND_COLORS[stint.compound] || '#888'}`,
                  }}
                >
                  <div className="stint-label">Stint {i + 1}</div>
                  <CompoundBadge compound={stint.compound} />
                </div>

                <div className="stint-controls">
                  <div className="form-group">
                    <label>Compound</label>
                    <select
                      value={stint.compound}
                      onChange={(e) =>
                        updateStint(i, 'compound', e.target.value)
                      }
                    >
                      {['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'].map(
                        (c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        )
                      )}
                    </select>
                  </div>

                  {i < stints.length - 1 ? (
                    <div className="form-group">
                      <label>Pit After Lap</label>
                      <input
                        type="number"
                        min={2}
                        max={raceInfo.total_laps - 1}
                        value={stint.pit_after ?? ''}
                        onChange={(e) =>
                          updateStint(i, 'pit_after', Number(e.target.value))
                        }
                      />
                    </div>
                  ) : (
                    <div className="form-group">
                      <label>End</label>
                      <div className="finish-label">
                        → Lap {raceInfo.total_laps}
                      </div>
                    </div>
                  )}

                  {stints.length > 1 && (
                    <button
                      className="btn-icon"
                      onClick={() => removeStint(i)}
                      title="Remove stint"
                    >
                      ✕
                    </button>
                  )}
                </div>

                {i < stints.length - 1 && (
                  <div className="pit-connector">
                    <div className="pit-line" />
                    <span className="pit-label">
                      PIT → Lap {stint.pit_after}
                    </span>
                    <div className="pit-line" />
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="stint-actions">
            <button className="btn btn-outline" onClick={addStint}>
              + Add Stint
            </button>
            <button className="btn btn-outline" onClick={loadActualStrategy}>
              📋 Load Actual
            </button>
            <button
              className="btn btn-primary"
              onClick={runSimulation}
              disabled={loading || !selectedDriver}
            >
              {loading ? (
                <>
                  <span className="spinner-small" /> Simulating…
                </>
              ) : (
                '▶ Run Simulation'
              )}
            </button>
          </div>
        </div>
      )}

      {/* ── Loading ── */}
      {loading && (
        <div className="card loading-card">
          <div className="loading-content">
            <div className="spinner" />
            <div className="loading-text">Running Strategy Simulation</div>
            <div className="loading-subtext">
              Predicting lap times for {selectedDriver}...
            </div>
          </div>
        </div>
      )}

      {/* ── Results ── */}
      {result && !loading && (
        <>
          {/* ── Hero result ── */}
          <div
            className={`card hero-card ${
              result.gain_loss <= 0 ? 'hero-faster' : 'hero-slower'
            }`}
          >
            <div className="hero-content">
              <div className="hero-delta">
                {formatDelta(result.gain_loss)}
              </div>
              <div className="hero-verdict">
                {result.gain_loss < 0
                  ? `${selectedDriver} would be FASTER`
                  : result.gain_loss > 0
                  ? `${selectedDriver} would be SLOWER`
                  : 'IDENTICAL to actual race'}
              </div>
              <div className="hero-meta">
                <span>Actual: {formatTime(result.actual_total)}</span>
                <span>•</span>
                <span>Simulated: {formatTime(result.sim_total)}</span>
                {result.mode.includes('VALIDATION') && (
                  <>
                    <span>•</span>
                    <span>MAE: {result.lap_mae.toFixed(3)}s</span>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* ── Strategy comparison ── */}
          <div className="strategy-comparison">
            <div className="card strategy-card">
              <div className="card-title">Actual Strategy</div>
              <div className="strategy-flow">
                {result.actual_strategy.compounds.map((c, i) => (
                  <React.Fragment key={i}>
                    <CompoundBadge compound={c} />
                    {i < result.actual_strategy.pit_laps.length && (
                      <span className="pit-arrow">
                        → L{result.actual_strategy.pit_laps[i]}
                      </span>
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>

            <div className="vs-divider">VS</div>

            <div className="card strategy-card">
              <div className="card-title">Your Strategy</div>
              <div className="strategy-flow">
                {result.sim_strategy.compounds.map((c, i) => (
                  <React.Fragment key={i}>
                    <CompoundBadge compound={c} />
                    {i < result.sim_strategy.pit_laps.length && (
                      <span className="pit-arrow">
                        → L{result.sim_strategy.pit_laps[i]}
                      </span>
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>

          {/* ── Tab navigation ── */}
          <div className="tab-nav">
            <button
              className={`tab-btn ${activeTab === 'chart' ? 'active' : ''}`}
              onClick={() => setActiveTab('chart')}
            >
              📊 Charts
            </button>
            <button
              className={`tab-btn ${activeTab === 'stints' ? 'active' : ''}`}
              onClick={() => setActiveTab('stints')}
            >
              🏁 Stints
            </button>
            <button
              className={`tab-btn ${activeTab === 'table' ? 'active' : ''}`}
              onClick={() => setActiveTab('table')}
            >
              📋 Lap Data
            </button>
          </div>

          {/* ── Chart tab ── */}
          {activeTab === 'chart' && (
            <>
              <div className="card">
                <div className="card-title">Lap Time Comparison</div>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart
                      data={chartData}
                      margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="rgba(255,255,255,0.06)"
                      />
                      <XAxis
                        dataKey="lap"
                        stroke="#555"
                        tick={{ fontSize: 11, fill: '#888' }}
                        label={{
                          value: 'Lap',
                          position: 'insideBottom',
                          offset: -10,
                          fill: '#888',
                        }}
                      />
                      <YAxis
                        stroke="#555"
                        tick={{ fontSize: 11, fill: '#888' }}
                        domain={['auto', 'auto']}
                        label={{
                          value: 'Time (s)',
                          angle: -90,
                          position: 'insideLeft',
                          fill: '#888',
                        }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: '#1a1a1a',
                          border: '1px solid #333',
                          borderRadius: 8,
                          fontSize: '0.85rem',
                        }}
                        formatter={(value: any, name: any): [string, string] => [
                          `${Number(value).toFixed(3)}s`,
                          name === 'actual' ? 'Actual' : 'Simulated',
                        ]}
                        labelFormatter={(label: any) => `Lap ${label}`}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: '0.8rem', color: '#888' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="actual"
                        stroke="#ffffff"
                        strokeWidth={2}
                        dot={false}
                        name="Actual"
                      />
                      <Line
                        type="monotone"
                        dataKey="simulated"
                        stroke="#e10600"
                        strokeWidth={2}
                        dot={false}
                        name="Simulated"
                        strokeDasharray="4 2"
                      />
                      {result.sim_strategy.pit_laps.map((p) => (
                        <ReferenceLine
                          key={`pit-${p}`}
                          x={p}
                          stroke="rgba(225,6,0,0.3)"
                          strokeDasharray="3 3"
                          label={{
                            value: `P`,
                            position: 'top',
                            fill: '#e10600',
                            fontSize: 10,
                          }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="card">
                <div className="card-title">Cumulative Time Delta</div>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={cumulativeData}
                      margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="rgba(255,255,255,0.06)"
                      />
                      <XAxis
                        dataKey="lap"
                        stroke="#555"
                        tick={{ fontSize: 11, fill: '#888' }}
                        label={{
                          value: 'Lap',
                          position: 'insideBottom',
                          offset: -10,
                          fill: '#888',
                        }}
                      />
                      <YAxis
                        stroke="#555"
                        tick={{ fontSize: 11, fill: '#888' }}
                        label={{
                          value: 'Δ Time (s)',
                          angle: -90,
                          position: 'insideLeft',
                          fill: '#888',
                        }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: '#1a1a1a',
                          border: '1px solid #333',
                          borderRadius: 8,
                          fontSize: '0.85rem',
                        }}
                        formatter={(value: any): [string, string] => [
                          formatDelta(Number(value)),
                          'Cumulative Δ',
                        ]}
                        labelFormatter={(label: any) => `Lap ${label}`}
                      />
                      <ReferenceLine y={0} stroke="#555" strokeDasharray="3 3" />
                      <Line
                        type="monotone"
                        dataKey="cumDiff"
                        stroke="#e10600"
                        strokeWidth={2}
                        dot={false}
                        name="Cumulative Delta"
                      />
                      {result.sim_strategy.pit_laps.map((p) => (
                        <ReferenceLine
                          key={`pit-cum-${p}`}
                          x={p}
                          stroke="rgba(225,6,0,0.3)"
                          strokeDasharray="3 3"
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}

          {/* ── Stints tab ── */}
          {activeTab === 'stints' && (
            <>
              <div className="card">
                <div className="card-title">Stint Breakdown</div>
                <div className="stint-table-container">
                  <table className="stint-table">
                    <thead>
                      <tr>
                        <th>Stint</th>
                        <th>Compound</th>
                        <th>Laps</th>
                        <th>Avg Actual</th>
                        <th>Avg Simulated</th>
                        <th>Stint Δ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.stint_summary.map((s) => (
                        <tr key={s.stint}>
                          <td className="stint-num">{s.stint}</td>
                          <td>
                            <CompoundBadge compound={s.compound} />
                          </td>
                          <td>
                            {s.start_lap}–{s.end_lap} ({s.laps})
                          </td>
                          <td>{s.avg_actual.toFixed(3)}s</td>
                          <td>{s.avg_sim.toFixed(3)}s</td>
                          <td
                            className={
                              s.stint_diff > 0.5
                                ? 'diff-positive'
                                : s.stint_diff < -0.5
                                ? 'diff-negative'
                                : 'diff-neutral'
                            }
                          >
                            {formatDelta(s.stint_diff)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {result.pit_summary.length > 0 && (
                <div className="card">
                  <div className="card-title">Pit Stop Comparison</div>
                  <div className="stint-table-container">
                    <table className="stint-table">
                      <thead>
                        <tr>
                          <th>Lap</th>
                          <th>Type</th>
                          <th>Compound</th>
                          <th>Actual</th>
                          <th>Simulated</th>
                          <th>Diff</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.pit_summary.map((p) => (
                          <tr key={p.lap}>
                            <td className="stint-num">{p.lap}</td>
                            <td>{p.event}</td>
                            <td>
                              <CompoundBadge compound={p.compound} />
                            </td>
                            <td>{p.actual.toFixed(3)}s</td>
                            <td>{p.simulated.toFixed(3)}s</td>
                            <td
                              className={
                                p.diff > 0
                                  ? 'diff-positive'
                                  : 'diff-negative'
                              }
                            >
                              {formatDelta(p.diff)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}

          {/* ── Table tab ── */}
          {activeTab === 'table' && (
            <div className="card">
              <div className="card-title">Lap-by-Lap Data</div>
              <div className="lap-table-container">
                <table className="lap-table">
                  <thead>
                    <tr>
                      <th>Lap</th>
                      <th>Compound</th>
                      <th>Event</th>
                      <th>Stint Lap</th>
                      <th>Actual</th>
                      <th>Simulated</th>
                      <th>Lap Δ</th>
                      <th>Cumul Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.laps.map((l) => (
                      <tr
                        key={l.lap}
                        className={
                          l.event.includes('pit')
                            ? 'pit-row'
                            : l.event === 'SC/VSC'
                            ? 'sc-row'
                            : ''
                        }
                      >
                        <td>{l.lap}</td>
                        <td>
                          <CompoundBadge compound={l.compound} />
                        </td>
                        <td className="event-cell">{l.event}</td>
                        <td className="stint-lap-cell">{l.stint_lap}</td>
                        <td>{l.actual.toFixed(3)}</td>
                        <td>{l.simulated.toFixed(3)}</td>
                        <td
                          className={
                            l.diff > 0.5
                              ? 'diff-positive'
                              : l.diff < -0.5
                              ? 'diff-negative'
                              : 'diff-neutral'
                          }
                        >
                          {formatDelta(l.diff)}
                        </td>
                        <td
                          className={
                            l.cumulative_diff > 1
                              ? 'diff-positive'
                              : l.cumulative_diff < -1
                              ? 'diff-negative'
                              : 'diff-neutral'
                          }
                        >
                          {formatDelta(l.cumulative_diff)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── Empty state ── */}
      {!result && !loading && !loadingRaces && (
        <div className="empty-state">
          <div className="empty-icon">🏎️</div>
          <div className="empty-title">SELECT A RACE AND BUILD YOUR STRATEGY</div>
          <div className="empty-subtitle">
            Choose a year, grand prix, and driver to begin
          </div>
        </div>
      )}

      {/* ── Footer ── */}
      <footer className="footer">
        <span>F1 Strategy Lab — Pure ML Prediction System</span>
        <span>XGBoost + ExtraTrees • No hacks, no rules</span>
      </footer>
    </div>
  );
}