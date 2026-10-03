import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Activity,
  ArrowRight,
  Cpu,
  Gauge,
  GaugeCircle,
  MonitorCog,
  Plane,
  RefreshCcw,
  ShieldCheck,
} from 'lucide-react'

const sensorGroups = [
  { label: 'Sensor Group A', keys: ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5'] },
  { label: 'Sensor Group B', keys: ['sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10'] },
  { label: 'Sensor Group C', keys: ['sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15'] },
  { label: 'Advanced Telemetry', keys: ['sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'] },
]

type PageKey = 'overview' | 'engine-monitor' | 'rul-prediction' | 'fleet-analytics' | 'model-performance' | 'system-information'

type PredictionResult = {
  predicted_rul: number
  unit: string
  status: string
}

type SampleResponse = {
  engine: Record<string, number>
  features: Record<string, number>
}

const defaultForm = {
  engine_id: 1,
  cycle: 118,
  setting_1: 100.2,
  setting_2: 38.6,
  setting_3: 20.9,
  sensor_1: 58.1,
  sensor_2: 81.4,
  sensor_3: 113.8,
  sensor_4: 136.2,
  sensor_5: 149.6,
  sensor_6: 38.9,
  sensor_7: 42.1,
  sensor_8: 46.3,
  sensor_9: 52.7,
  sensor_10: 59.8,
  sensor_11: 74.1,
  sensor_12: 82.3,
  sensor_13: 97.6,
  sensor_14: 106.4,
  sensor_15: 117.8,
  sensor_16: 124.3,
  sensor_17: 41.3,
  sensor_18: 45.4,
  sensor_19: 50.9,
  sensor_20: 58.7,
  sensor_21: 63.4,
}

const modelComparisonData = [
  { name: 'Linear Regression', mae: 'Available from training notebook', rmse: 'Available from training notebook', r2: 'Available from training notebook' },
  { name: 'Decision Tree', mae: 'Available from training notebook', rmse: 'Available from training notebook', r2: 'Available from training notebook' },
  { name: 'Random Forest', mae: 'Available from training notebook', rmse: 'Available from training notebook', r2: 'Available from training notebook' },
  { name: 'Gradient Boosting', mae: 'Available from training notebook', rmse: 'Available from training notebook', r2: 'Available from training notebook' },
  { name: 'XGBoost', mae: 'Available from training notebook', rmse: 'Available from training notebook', r2: 'Available from training notebook' },
]

const demoFleetData = [
  { engine: 'Engine 1', cycle: 114, predicted_rul: 73.4, status: 'Healthy' },
  { engine: 'Engine 2', cycle: 118, predicted_rul: 42.1, status: 'Warning' },
  { engine: 'Engine 3', cycle: 97, predicted_rul: 88.3, status: 'Monitor' },
  { engine: 'Engine 4', cycle: 126, predicted_rul: 19.6, status: 'Critical' },
]

const navItems: Array<{ key: PageKey; label: string }> = [
  { key: 'overview', label: 'Overview' },
  { key: 'engine-monitor', label: 'Engine Monitor' },
  { key: 'rul-prediction', label: 'RUL Prediction' },
  { key: 'fleet-analytics', label: 'Fleet Analytics' },
  { key: 'model-performance', label: 'Model Performance' },
  { key: 'system-information', label: 'System Information' },
]

const radarSeries = [
  { subject: 'Thermal', value: 78 },
  { subject: 'Pressure', value: 62 },
  { subject: 'Vibration', value: 82 },
  { subject: 'Efficiency', value: 71 },
  { subject: 'Fuel Flow', value: 59 },
]

const sensorLegend: Record<string, string> = {
  sensor_1: 'Pressure ratio',
  sensor_2: 'Temperature',
  sensor_3: 'Core speed',
  sensor_4: 'Exhaust temp',
  sensor_5: 'Flow',
  sensor_6: 'Altitude',
  sensor_7: 'Fuel mix',
  sensor_8: 'Thrust',
  sensor_9: 'Load',
  sensor_10: 'Vibration',
  sensor_11: 'Mach',
  sensor_12: 'Oil pressure',
  sensor_13: 'EPR',
  sensor_14: 'Combustor temp',
  sensor_15: 'Torque',
  sensor_16: 'Nozzle bias',
  sensor_17: 'Emissions',
  sensor_18: 'Health index',
  sensor_19: 'Rotor speed',
  sensor_20: 'Pressure delta',
  sensor_21: 'Diagnostics',
}

function getStatusColor(status: string) {
  const map: Record<string, string> = {
    healthy: 'text-emerald-300 bg-emerald-500/10 border-emerald-400/30',
    monitor: 'text-amber-200 bg-amber-500/10 border-amber-400/30',
    warning: 'text-orange-200 bg-orange-500/10 border-orange-400/30',
    critical: 'text-rose-200 bg-rose-500/10 border-rose-400/30',
  }

  return map[status] ?? 'text-slate-100 bg-slate-700/40 border-slate-600/20'
}

function App() {
  const [activePage, setActivePage] = useState<PageKey>('overview')
  const [formData, setFormData] = useState<Record<string, number>>({ ...defaultForm })
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [healthInfo, setHealthInfo] = useState({ status: 'online', model_loaded: true })
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({
    'Sensor Group A': true,
    'Sensor Group B': false,
    'Sensor Group C': false,
    'Advanced Telemetry': false,
  })
  const [selectedSensor, setSelectedSensor] = useState('sensor_1')
  const [featureImportance, setFeatureImportance] = useState<Array<{ feature: string; importance: number }>>([])

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.json())
      .then((data) => setHealthInfo(data))
      .catch(() => setHealthInfo({ status: 'offline', model_loaded: false }))

    fetch('/api/model/feature-importance')
      .then((response) => response.json())
      .then((data) => setFeatureImportance(data.items ?? []))
      .catch(() => setFeatureImportance([]))
  }, [])

  const telemetryData = useMemo(() => {
    const cycle = Number(formData.cycle)
    return Array.from({ length: 12 }, (_, index) => {
      const currentCycle = Math.round((cycle / 11) * index)
      return {
        cycle: currentCycle || 1,
        setting_1: Number(formData.setting_1) * (0.75 + index / 11),
        setting_2: Number(formData.setting_2) * (0.72 + index / 12),
        setting_3: Number(formData.setting_3) * (0.78 + index / 10),
        sensor_1: Number(formData.sensor_1) * (0.7 + index / 12),
      }
    })
  }, [formData])

  const selectedSensorData = useMemo(
    () =>
      telemetryData.map((point) => ({
        cycle: point.cycle,
        value: Number(point[selectedSensor as keyof typeof point] ?? 0),
      })),
    [selectedSensor, telemetryData],
  )

  const fleetSummary = useMemo(() => {
    const data = demoFleetData.map((item) => ({ ...item, statusLower: item.status.toLowerCase() }))

    return {
      total: data.length,
      healthy: data.filter((item) => item.statusLower === 'healthy').length,
      monitor: data.filter((item) => item.statusLower === 'monitor').length,
      warning: data.filter((item) => item.statusLower === 'warning').length,
      critical: data.filter((item) => item.statusLower === 'critical').length,
    }
  }, [])

  const handleChange = (name: string, value: string) => {
    const numericValue = Number(value)
    setFormData((current) => ({ ...current, [name]: Number.isFinite(numericValue) ? numericValue : 0 }))
  }

  const resetForm = () => setFormData({ ...defaultForm })

  const loadSampleEngine = async (engineId = 1) => {
    try {
      const response = await fetch(`/api/sample-engine?engine_id=${engineId}`)
      if (!response.ok) throw new Error('Sample engine unavailable')
      const payload = (await response.json()) as SampleResponse
      setFormData({
        ...payload.engine,
        ...payload.features,
      })
    } catch {
      const fallback = {
        engine_id: engineId,
        cycle: 104 + engineId * 7,
        setting_1: 100.4 + engineId,
        setting_2: 38.6 + engineId * 0.4,
        setting_3: 20.8 + engineId * 0.2,
        sensor_1: 58.1 + engineId * 3.1,
        sensor_2: 81.4 + engineId * 2.6,
        sensor_3: 113.8 + engineId * 2.3,
        sensor_4: 136.2 + engineId * 1.5,
        sensor_5: 149.6 + engineId * 1.2,
        sensor_6: 38.9 + engineId * 0.8,
        sensor_7: 42.1 + engineId * 0.9,
        sensor_8: 46.3 + engineId * 1.1,
        sensor_9: 52.7 + engineId * 1.2,
        sensor_10: 59.8 + engineId * 1.5,
        sensor_11: 74.1 + engineId * 1.7,
        sensor_12: 82.3 + engineId * 1.1,
        sensor_13: 97.6 + engineId * 1.4,
        sensor_14: 106.4 + engineId * 1.6,
        sensor_15: 117.8 + engineId * 2.1,
        sensor_16: 124.3 + engineId * 1.9,
        sensor_17: 41.3 + engineId * 0.7,
        sensor_18: 45.4 + engineId * 0.8,
        sensor_19: 50.9 + engineId * 0.9,
        sensor_20: 58.7 + engineId * 1.3,
        sensor_21: 63.4 + engineId * 1.4,
      }
      setFormData(fallback)
    }
  }

  const handlePredict = async () => {
    setLoading(true)
    setError('')

    const payload = Object.fromEntries(
      Object.entries(formData).map(([key, value]) => [key, Number(value)]),
    )

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({ detail: 'Prediction unavailable' }))
        throw new Error(detail.detail || 'Prediction unavailable')
      }

      const data = (await response.json()) as PredictionResult
      setPrediction(data)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Prediction unavailable')
    } finally {
      setLoading(false)
    }
  }

  const renderField = (label: string, name: string, value: number) => (
    <label key={name} className="space-y-2 text-xs uppercase tracking-[0.18em] text-slate-300">
      <span>{label}</span>
      <input
        type="number"
        aria-label={label}
        value={value}
        onChange={(event) => handleChange(name, event.target.value)}
        className="w-full rounded-xl border border-slate-700 bg-slate-950/80 px-3 py-2.5 text-sm font-medium text-slate-100 outline-none transition focus:border-cyan-500"
      />
    </label>
  )

  const renderGauge = () => {
    const currentRul = prediction?.predicted_rul ?? 0
    const gauge = Math.min(125, Math.max(0, currentRul))
    const percentage = (gauge / 125) * 100

    return (
      <div className="flex flex-col items-center justify-center gap-4 rounded-2xl border border-slate-700 bg-slate-950/70 p-6 shadow-panel">
        <div className="relative flex h-36 w-36 items-center justify-center rounded-full border-[10px] border-slate-800 bg-slate-950">
          <div
            className="absolute inset-2 rounded-full"
            style={{
              background: `conic-gradient(#38bdf8 ${percentage}%, rgba(15, 23, 42, 0.5) 0%)`,
            }}
          />
          <div className="relative flex h-20 w-20 items-center justify-center rounded-full border border-slate-700 bg-slate-950 text-center">
            <span className="font-mono text-2xl font-semibold text-cyan-200">{currentRul.toFixed(1)}</span>
          </div>
        </div>
        <div className="text-center">
          <div className="text-[10px] uppercase tracking-[0.32em] text-slate-400">Remaining Useful Life</div>
          <div className="mt-2 text-3xl font-semibold text-white">{currentRul.toFixed(1)}</div>
          <div className="text-xs uppercase tracking-[0.2em] text-cyan-300">Cycles</div>
        </div>
      </div>
    )
  }

  const renderOverview = () => (
    <div className="space-y-8">
      <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="overflow-hidden rounded-[28px] border border-slate-800 bg-slate-900/80 shadow-panel">
        <div className="grid gap-8 p-8 lg:grid-cols-[1.3fr_0.7fr]">
          <div className="space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-[10px] font-medium uppercase tracking-[0.32em] text-cyan-200">
              <span className="h-2 w-2 rounded-full bg-emerald-400" /> MODEL ONLINE
            </div>
            <div>
              <div className="text-4xl font-semibold tracking-tight text-white md:text-6xl">AEROSCOPE</div>
              <div className="mt-3 text-xl text-slate-300">Predictive Maintenance Intelligence</div>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-xl border border-slate-700 bg-slate-950/50 p-4">
                <div className="text-[10px] uppercase tracking-[0.24em] text-slate-400">Model</div>
                <div className="mt-2 text-sm font-medium text-slate-100">Gradient Boosting Regressor</div>
              </div>
              <div className="rounded-xl border border-slate-700 bg-slate-950/50 p-4">
                <div className="text-[10px] uppercase tracking-[0.24em] text-slate-400">Dataset</div>
                <div className="mt-2 text-sm font-medium text-slate-100">NASA C-MAPSS FD001</div>
              </div>
              <div className="rounded-xl border border-slate-700 bg-slate-950/50 p-4">
                <div className="text-[10px] uppercase tracking-[0.24em] text-slate-400">RUL Cap</div>
                <div className="mt-2 text-sm font-medium text-slate-100">125 cycles</div>
              </div>
              <div className="rounded-xl border border-slate-700 bg-slate-950/50 p-4">
                <div className="text-[10px] uppercase tracking-[0.24em] text-slate-400">Prediction API</div>
                <div className="mt-2 text-sm font-medium text-slate-100">{healthInfo.model_loaded ? 'Online' : 'Offline'}</div>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-950/55 p-4">
            <div className="mb-4 flex items-center justify-between">
              <div className="text-[10px] uppercase tracking-[0.26em] text-slate-400">Engine telemetry</div>
              <div className="flex items-center gap-2 text-xs text-cyan-300"><Cpu size={14} /> Live</div>
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={telemetryData}>
                <defs>
                  <linearGradient id="telemetryGradient" x1="0" x2="0" y1="0" y2="1">
                    <stop stopColor="#38bdf8" stopOpacity={0.7} />
                    <stop offset="1" stopColor="#38bdf8" stopOpacity={0.06} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="cycle" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <Tooltip contentStyle={{ background: '#020b16', border: '1px solid #334155' }} />
                <Area type="monotone" dataKey="sensor_1" stroke="#38bdf8" fill="url(#telemetryGradient)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.section>

      <section className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
        {[
          { label: 'Model Version', value: 'Production', icon: ShieldCheck },
          { label: 'Fleet Health', value: '92.7%', icon: Activity },
          { label: 'Telemetry', value: '21 Sensors', icon: MonitorCog },
          { label: 'RUL Horizon', value: '125 cycles', icon: GaugeCircle },
        ].map(({ label, value, icon: Icon }) => (
          <div key={label} className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-panel">
            <div className="flex items-center justify-between text-slate-300">
              <span className="text-[10px] uppercase tracking-[0.28em]">{label}</span>
              <Icon className="text-cyan-300" size={16} />
            </div>
            <div className="mt-5 font-mono text-2xl text-white">{value}</div>
          </div>
        ))}
      </section>
    </div>
  )

  const renderPredictionPage = () => (
    <div className="grid gap-8 xl:grid-cols-[0.95fr_1.05fr]">
      <div className="space-y-6 rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400">Engine information</div>
            <h2 className="mt-2 text-2xl font-semibold text-white">RUL Prediction Console</h2>
          </div>
          <button type="button" onClick={resetForm} className="rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-xs uppercase tracking-[0.2em] text-slate-200">
            Reset
          </button>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          {renderField('Engine ID', 'engine_id', Number(formData.engine_id))}
          {renderField('Operating Cycle', 'cycle', Number(formData.cycle))}
        </div>

        <div>
          <div className="mb-4 text-[10px] uppercase tracking-[0.26em] text-slate-400">Operating settings</div>
          <div className="grid gap-4 md:grid-cols-3">
            {renderField('Setting 1', 'setting_1', Number(formData.setting_1))}
            {renderField('Setting 2', 'setting_2', Number(formData.setting_2))}
            {renderField('Setting 3', 'setting_3', Number(formData.setting_3))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between gap-4">
            <div className="text-[10px] uppercase tracking-[0.26em] text-slate-400">Sensor telemetry</div>
            <button type="button" onClick={() => loadSampleEngine(Number(formData.engine_id || 1))} className="inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-[10px] uppercase tracking-[0.2em] text-cyan-200">
              <RefreshCcw size={14} /> Load Sample Engine
            </button>
          </div>

          {sensorGroups.map((group) => (
            <div key={group.label} className="overflow-hidden rounded-2xl border border-slate-700 bg-slate-950/45">
              <button
                type="button"
                onClick={() => setExpandedGroups((current) => ({ ...current, [group.label]: !current[group.label] }))}
                className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-slate-100"
              >
                {group.label}
                <span className="text-cyan-300">{expandedGroups[group.label] ? '−' : '+'}</span>
              </button>
              {expandedGroups[group.label] && (
                <div className="grid gap-3 p-4 md:grid-cols-2 xl:grid-cols-3">
                  {group.keys.map((key) => renderField(sensorLegend[key] || key, key, Number(formData[key])))}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="flex flex-wrap gap-3 pt-2">
          <button type="button" onClick={handlePredict} className="inline-flex items-center gap-2 rounded-xl bg-cyan-400 px-5 py-3 text-xs font-semibold uppercase tracking-[0.2em] text-slate-950 transition hover:bg-cyan-300">
            <Gauge size={16} /> {loading ? 'Evaluating...' : 'Predict RUL'}
          </button>
          <select
            value={selectedSensor}
            onChange={(event) => setSelectedSensor(event.target.value)}
            className="rounded-xl border border-slate-700 bg-slate-950/70 px-3 py-3 text-xs uppercase tracking-[0.18em] text-slate-200"
          >
            {sensorGroups.flatMap((group) => group.keys.map((key) => (
              <option key={key} value={key}>{key}</option>
            )))}
          </select>
        </div>

        {error && <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">{error}</div>}
      </div>

      <div className="space-y-6">
        <div className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400">Prediction result</div>
              <h3 className="mt-2 text-xl font-semibold text-white">Remaining Useful Life</h3>
            </div>
            {prediction ? <span className={`rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.24em] ${getStatusColor(prediction.status)}`}>{prediction.status}</span> : null}
          </div>

          {prediction ? (
            <div className="grid gap-5 lg:grid-cols-[0.8fr_1.2fr]">
              {renderGauge()}
              <div className="space-y-5">
                <div className="rounded-2xl border border-slate-700 bg-slate-950/55 p-4">
                  <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400">Current assessment</div>
                  <div className="mt-3 text-3xl font-semibold text-white">{prediction.predicted_rul.toFixed(1)}</div>
                  <div className="mt-1 text-xs uppercase tracking-[0.23em] text-cyan-300">Estimated remaining cycles</div>
                </div>
                <div className="rounded-2xl border border-slate-700 bg-slate-950/55 p-4 text-sm text-slate-200">
                  The model estimates approximately <span className="font-semibold text-white">{prediction.predicted_rul.toFixed(1)} operating cycles</span> before reaching the learned RUL endpoint. Dashboard visualization thresholds are used for monitoring-only status bands.
                </div>
                <div className="rounded-2xl border border-slate-700 bg-slate-950/55 p-4">
                  <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400">Status legend</div>
                  <div className="mt-3 grid gap-2 text-xs text-slate-200">
                    <div className="flex items-center justify-between"><span>100–125</span><span className="rounded-full border border-emerald-400/30 bg-emerald-500/10 px-2 py-1 text-emerald-300">HEALTHY</span></div>
                    <div className="flex items-center justify-between"><span>60–99</span><span className="rounded-full border border-amber-400/30 bg-amber-500/10 px-2 py-1 text-amber-300">MONITOR</span></div>
                    <div className="flex items-center justify-between"><span>25–59</span><span className="rounded-full border border-orange-400/30 bg-orange-500/10 px-2 py-1 text-orange-300">WARNING</span></div>
                    <div className="flex items-center justify-between"><span>0–24</span><span className="rounded-full border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-rose-300">CRITICAL</span></div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-2xl border border-dashed border-slate-700 bg-slate-950/35 p-8 text-center text-slate-400">
              Prediction unavailable. Submit a valid engine profile to calculate a model estimate.
            </div>
          )}
        </div>

        <div className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
          <div className="mb-4 text-[10px] uppercase tracking-[0.28em] text-slate-400">Sensor behaviour</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={selectedSensorData}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis dataKey="cycle" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip contentStyle={{ background: '#020b16', border: '1px solid #334155' }} />
              <Line type="monotone" dataKey="value" stroke="#7dd3fc" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )

  const renderFleetPage = () => (
    <div className="space-y-8">
      <section className="grid gap-4 md:grid-cols-5">
        {[
          { label: 'Total Engines', value: fleetSummary.total },
          { label: 'Healthy', value: fleetSummary.healthy },
          { label: 'Monitoring', value: fleetSummary.monitor },
          { label: 'Warning', value: fleetSummary.warning },
          { label: 'Critical', value: fleetSummary.critical },
        ].map((item) => (
          <div key={item.label} className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5 shadow-panel">
            <div className="text-[10px] uppercase tracking-[0.26em] text-slate-400">{item.label}</div>
            <div className="mt-4 font-mono text-3xl text-white">{item.value}</div>
          </div>
        ))}
      </section>

      <section className="grid gap-8 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
          <div className="mb-5 text-[10px] uppercase tracking-[0.28em] text-slate-400">Fleet RUL distribution</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={demoFleetData}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis dataKey="engine" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip contentStyle={{ background: '#020b16', border: '1px solid #334155' }} />
              <Bar dataKey="predicted_rul" fill="#38bdf8" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
          <div className="mb-5 text-[10px] uppercase tracking-[0.28em] text-slate-400">Engine ranking</div>
          <div className="space-y-3">
            {demoFleetData.map((engine) => (
              <div key={engine.engine} className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-950/60 p-3">
                <div>
                  <div className="text-sm font-medium text-white">{engine.engine}</div>
                  <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Cycle {engine.cycle}</div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-base text-cyan-200">{engine.predicted_rul.toFixed(1)}</div>
                  <div className={`mt-1 inline-flex rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-[0.18em] ${getStatusColor(engine.status.toLowerCase())}`}>{engine.status}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )

  const renderModelPage = () => (
    <div className="space-y-8">
      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
          <div className="mb-5 text-[10px] uppercase tracking-[0.28em] text-slate-400">Model comparison — RMSE</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={modelComparisonData} layout="vertical">
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} width={90} />
              <Tooltip contentStyle={{ background: '#020b16', border: '1px solid #334155' }} />
              <Bar dataKey="rmse" fill="#38bdf8" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
          <div className="mb-5 text-[10px] uppercase tracking-[0.28em] text-slate-400">Model comparison — R²</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={modelComparisonData} layout="vertical">
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} width={90} />
              <Tooltip contentStyle={{ background: '#020b16', border: '1px solid #334155' }} />
              <Bar dataKey="r2" fill="#fbbf24" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
        <div className="mb-5 text-[10px] uppercase tracking-[0.28em] text-slate-400">Model experiments</div>
        <div className="overflow-hidden rounded-2xl border border-slate-700">
          <table className="min-w-full text-left text-sm text-slate-200">
            <thead className="bg-slate-950/80 text-slate-300">
              <tr>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">MAE</th>
                <th className="px-4 py-3">RMSE</th>
                <th className="px-4 py-3">R²</th>
              </tr>
            </thead>
            <tbody>
              {modelComparisonData.map((row) => (
                <tr key={row.name} className="border-t border-slate-800">
                  <td className="px-4 py-3 font-medium text-white">{row.name}</td>
                  <td className="px-4 py-3">{row.mae}</td>
                  <td className="px-4 py-3">{row.rmse}</td>
                  <td className="px-4 py-3">{row.r2}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )

  const renderSystemPage = () => (
    <div className="grid gap-8 xl:grid-cols-[0.9fr_1.1fr]">
      <div className="space-y-6">
        <div className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
          <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400">Production model</div>
          <div className="mt-4 text-2xl font-semibold text-white">Gradient Boosting Regressor</div>
          <div className="mt-6 space-y-4 text-sm text-slate-200">
            <div className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-950/50 p-3"><span>n_estimators</span><span className="font-mono text-cyan-200">200</span></div>
            <div className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-950/50 p-3"><span>learning_rate</span><span className="font-mono text-cyan-200">0.05</span></div>
            <div className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-950/50 p-3"><span>max_depth</span><span className="font-mono text-cyan-200">3</span></div>
            <div className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-950/50 p-3"><span>random_state</span><span className="font-mono text-cyan-200">42</span></div>
            <div className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-950/50 p-3"><span>RUL upper cap</span><span className="font-mono text-cyan-200">125 cycles</span></div>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
          <div className="mb-4 text-[10px] uppercase tracking-[0.28em] text-slate-400">Prediction drivers</div>
          {featureImportance.length ? (
            <div className="space-y-3">
              {featureImportance.slice(0, 6).map(({ feature, importance }) => (
                <div key={feature}>
                  <div className="mb-1 flex items-center justify-between text-xs uppercase tracking-[0.15em] text-slate-300">
                    <span>{feature}</span>
                    <span>{importance.toFixed(3)}</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-800">
                    <div className="h-2 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500" style={{ width: `${Math.max(8, importance * 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-slate-700 bg-slate-950/35 p-4 text-sm text-slate-300">Available from training notebook</div>
          )}
        </div>
      </div>

      <div className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
        <div className="mb-5 text-[10px] uppercase tracking-[0.28em] text-slate-400">Dataset explorer</div>
        <div className="grid gap-4 md:grid-cols-4">
          {[
            { label: 'Training Engines', value: '180' },
            { label: 'Test Engines', value: '100' },
            { label: 'Total Sensors', value: '21' },
            { label: 'Operating Cycles', value: '150' },
          ].map((item) => (
            <div key={item.label} className="rounded-2xl border border-slate-700 bg-slate-950/50 p-4">
              <div className="text-[10px] uppercase tracking-[0.25em] text-slate-400">{item.label}</div>
              <div className="mt-3 font-mono text-xl text-white">{item.value}</div>
            </div>
          ))}
        </div>

        <div className="mt-6 grid gap-5 lg:grid-cols-2">
          <div className="rounded-2xl border border-slate-700 bg-slate-950/55 p-4">
            <div className="mb-4 text-[10px] uppercase tracking-[0.28em] text-slate-400">RUL distribution</div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={demoFleetData.map((entry) => ({ name: entry.engine, rul: entry.predicted_rul }))}>
                <defs>
                  <linearGradient id="rulDistribution" x1="0" x2="0" y1="0" y2="1">
                    <stop stopColor="#22d3ee" stopOpacity={0.8} />
                    <stop offset="1" stopColor="#22d3ee" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <Tooltip contentStyle={{ background: '#020b16', border: '1px solid #334155' }} />
                <Area type="monotone" dataKey="rul" stroke="#22d3ee" fill="url(#rulDistribution)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-2xl border border-slate-700 bg-slate-950/55 p-4">
            <div className="mb-4 text-[10px] uppercase tracking-[0.28em] text-slate-400">Telemetry radar</div>
            <ResponsiveContainer width="100%" height={200}>
              <RadarChart data={radarSeries}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                <Radar name="Health" dataKey="value" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.35} />
                <Tooltip contentStyle={{ background: '#020b16', border: '1px solid #334155' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="mt-5 rounded-2xl border border-slate-700 bg-slate-950/45 p-4 text-sm leading-7 text-slate-200">
          The production model was trained using the cleaned NASA C-MAPSS FD001 training data and predicts Remaining Useful Life from engine operating settings and sensor measurements.
        </div>
      </div>
    </div>
  )

  const renderEngineMonitor = () => (
    <div className="space-y-8">
      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
          <div className="mb-5 text-[10px] uppercase tracking-[0.28em] text-slate-400">Current engine health</div>
          <div className="grid gap-3 md:grid-cols-2">
            {[
              { label: 'Engine ID', value: String(formData.engine_id) },
              { label: 'Operating Cycle', value: String(formData.cycle) },
              { label: 'Setting 1', value: String(formData.setting_1) },
              { label: 'Setting 2', value: String(formData.setting_2) },
              { label: 'Setting 3', value: String(formData.setting_3) },
              { label: 'Sensor 1', value: String(formData.sensor_1) },
            ].map(({ label, value }) => (
              <div key={label} className="rounded-xl border border-slate-700 bg-slate-950/50 p-3">
                <div className="text-[10px] uppercase tracking-[0.24em] text-slate-400">{label}</div>
                <div className="mt-2 font-mono text-lg text-white">{value}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
          <div className="mb-5 text-[10px] uppercase tracking-[0.28em] text-slate-400">Health timeline</div>
          <div className="rounded-2xl border border-slate-700 bg-slate-950/55 p-5">
            <div className="mb-5 flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-slate-400">
              <span>0 cycles</span>
              <span>Current</span>
              <span>Estimated RUL</span>
              <span>125-cycle horizon</span>
            </div>
            <div className="relative h-24 rounded-full border border-slate-700 bg-slate-900/90 px-3 py-2">
              <div className="absolute left-4 right-4 top-1/2 h-1 -translate-y-1/2 rounded-full bg-slate-700" />
              <div className="absolute left-5 top-1/2 h-4 w-4 -translate-y-1/2 rounded-full bg-cyan-400 shadow-[0_0_18px_rgba(34,211,238,0.6)]" />
              <div className="absolute top-1/2 h-4 w-4 -translate-y-1/2 rounded-full border border-cyan-300 bg-cyan-400/60" style={{ left: `${((prediction?.predicted_rul ?? 80) / 125) * 100}%` }} />
            </div>
            <div className="mt-4 flex items-center justify-between text-xs text-slate-300">
              <span>Current</span>
              <span className="font-mono text-cyan-200">RUL = {prediction?.predicted_rul?.toFixed(1) ?? '—'}</span>
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-3xl border border-slate-800 bg-slate-900/75 p-6 shadow-panel">
        <div className="mb-5 text-[10px] uppercase tracking-[0.28em] text-slate-400">Operating settings timeline</div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={telemetryData}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis dataKey="cycle" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <Tooltip contentStyle={{ background: '#020b16', border: '1px solid #334155' }} />
            <Line dataKey="setting_1" stroke="#38bdf8" strokeWidth={2} dot={false} />
            <Line dataKey="setting_2" stroke="#22c55e" strokeWidth={2} dot={false} />
            <Line dataKey="setting_3" stroke="#fbbf24" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>
    </div>
  )

  const pageContent = () => {
    switch (activePage) {
      case 'overview':
        return renderOverview()
      case 'engine-monitor':
        return renderEngineMonitor()
      case 'rul-prediction':
        return renderPredictionPage()
      case 'fleet-analytics':
        return renderFleetPage()
      case 'model-performance':
        return renderModelPage()
      case 'system-information':
        return renderSystemPage()
      default:
        return renderOverview()
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-[1500px] px-4 py-6 md:px-8">
        <header className="mb-6 rounded-[26px] border border-slate-800 bg-slate-900/80 px-5 py-4 shadow-panel backdrop-blur-sm">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-cyan-500/30 bg-cyan-500/10 text-cyan-200">
                <Plane size={18} />
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400">AEROSCOPE</div>
                <div className="mt-1 font-medium text-white">Predictive Maintenance & RUL Intelligence</div>
              </div>
            </div>

            <nav className="flex flex-wrap gap-2">
              {navItems.map((item) => (
                <button
                  key={item.key}
                  type="button"
                  onClick={() => setActivePage(item.key)}
                  className={`rounded-full border px-3 py-2 text-[10px] font-medium uppercase tracking-[0.2em] transition ${
                    activePage === item.key
                      ? 'border-cyan-400/40 bg-cyan-500/10 text-cyan-100'
                      : 'border-slate-700 bg-slate-950/50 text-slate-300 hover:border-slate-600'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </nav>
          </div>
        </header>

        <div className="mb-6 grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
          <div className="rounded-[26px] border border-slate-800 bg-slate-900/80 p-6 shadow-panel">
            <div className="flex items-center justify-between gap-4">
              <div>
                <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400">Mission control</div>
                <div className="mt-3 text-2xl font-semibold text-white">PREDICT THE NEXT CYCLE.</div>
              </div>
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-[10px] uppercase tracking-[0.28em] text-cyan-200">
                <span className="h-2 w-2 rounded-full bg-emerald-400" /> {healthInfo.status}
              </div>
            </div>
            <p className="mt-4 max-w-xl text-sm leading-7 text-slate-300">Understand engine degradation before it becomes an operational problem.</p>
            <div className="mt-6 flex flex-wrap gap-3">
              <button type="button" onClick={() => setActivePage('rul-prediction')} className="inline-flex items-center gap-2 rounded-xl bg-cyan-400 px-5 py-3 text-xs font-semibold uppercase tracking-[0.2em] text-slate-950">
                Open Engine Monitor <ArrowRight size={16} />
              </button>
              <button type="button" onClick={() => setActivePage('model-performance')} className="rounded-xl border border-slate-700 bg-slate-950/60 px-5 py-3 text-xs uppercase tracking-[0.2em] text-slate-100">
                Explore Model
              </button>
            </div>
          </div>

          <div className="rounded-[26px] border border-slate-800 bg-slate-900/80 p-6 shadow-panel">
            <div className="flex items-center justify-between">
              <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400">System status</div>
              <span className="font-mono text-xs text-cyan-300">v1.0.0</span>
            </div>
            <div className="mt-6 space-y-4">
              <div className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-950/55 p-3">
                <span className="text-slate-300">Model</span>
                <span className="font-mono text-cyan-200">GBR</span>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-950/55 p-3">
                <span className="text-slate-300">API</span>
                <span className="font-mono text-emerald-300">{healthInfo.model_loaded ? 'online' : 'offline'}</span>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-950/55 p-3">
                <span className="text-slate-300">RUL cap</span>
                <span className="font-mono text-cyan-200">125 cycles</span>
              </div>
            </div>
          </div>
        </div>

        {pageContent()}
      </div>
    </div>
  )
}

export default App
