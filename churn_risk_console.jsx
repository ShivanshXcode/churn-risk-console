import React, { useState, useMemo } from "react";
import { ArrowUp, ArrowDown, Activity } from "lucide-react";

// ---------------------------------------------------------------------------
// Model exported from a logistic regression trained on 5,000 synthetic
// customer records (see analysis.py / model_export.json). Test AUC: 0.788
// ---------------------------------------------------------------------------
const MODEL = {
  feature_order: [
    "age", "partner", "dependents", "tenure_months", "tech_support",
    "online_security", "streaming_service", "paperless_billing",
    "monthly_charges", "avg_monthly_usage_gb", "support_calls_6mo",
    "late_payments_12mo", "satisfaction_score", "contract_One year",
    "contract_Two year", "internet_service_Fiber optic", "internet_service_No",
    "payment_method_Credit card", "payment_method_Electronic check",
    "payment_method_Mailed check",
  ],
  coefficients: [
    -0.022180053091477553, 0.027033431402926144, 0.02796121453173792,
    -0.4574936766757718, -0.17436036228508794, -0.24074915979272732,
    -0.03873531304236974, 0.02928290803203839, 0.22566656441544636,
    -0.12092004977951178, 0.3736170250632281, 0.15286636992452718,
    -0.7543999952000309, -0.6255917238267444, -0.8351900941292001,
    0.283286293468057, -0.11405808688725556, -0.033644738287762425,
    0.16687102612850113, -0.059334917677688864,
  ],
  intercept: -1.089900884192974,
  scaler_mean: [
    42.684266666666666, 0.48746666666666666, 0.30906666666666666, 24.1656,
    0.40026666666666666, 0.3752, 0.44506666666666667, 0.6008,
    62.24523466666667, 79.59744, 1.2597333333333334, 0.5925333333333334,
    3.0370666666666666, 0.26, 0.18693333333333334, 0.4432,
    0.21333333333333335, 0.2541333333333333, 0.3389333333333333, 0.1816,
  ],
  scaler_scale: [
    13.583599638125708, 0.49984289087227757, 0.46210871255822716,
    16.173279711919903, 0.4899523060688889, 0.4841745139926305,
    0.4969731671719198, 0.4897339685992794, 26.215141374116822,
    55.227758323567684, 1.134080506646488, 0.7647903300179003,
    1.4125011134705543, 0.43863424398922624, 0.3898580026397075,
    0.49676328366738215, 0.4096611065529924, 0.43537292316153764,
    0.4733471547277842, 0.3855145133454771,
  ],
  test_auc: 0.7881,
};

const COLORS = {
  bg: "#12181C",
  panel: "#1B242A",
  panel2: "#20292F",
  border: "#2A363D",
  text: "#E8EDEF",
  muted: "#8B9AA3",
  low: "#3FAE8C",
  medium: "#D9A441",
  high: "#E8823D",
  critical: "#C4574A",
};

const FEATURE_LABELS = {
  age: "Age",
  partner: "Has partner",
  dependents: "Has dependents",
  tenure_months: "Tenure (months)",
  tech_support: "Tech support add-on",
  online_security: "Online security add-on",
  streaming_service: "Streaming add-on",
  paperless_billing: "Paperless billing",
  monthly_charges: "Monthly charges",
  avg_monthly_usage_gb: "Avg. monthly usage",
  support_calls_6mo: "Support calls (6mo)",
  late_payments_12mo: "Late payments (12mo)",
  satisfaction_score: "Satisfaction score",
  "contract_One year": "One-year contract",
  "contract_Two year": "Two-year contract",
  "internet_service_Fiber optic": "Fiber internet",
  "internet_service_No": "No internet service",
  "payment_method_Credit card": "Pays by credit card",
  "payment_method_Electronic check": "Pays by e-check",
  "payment_method_Mailed check": "Pays by mailed check",
};

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function tierFor(prob) {
  if (prob < 0.25) return { label: "Low", color: COLORS.low };
  if (prob < 0.45) return { label: "Medium", color: COLORS.medium };
  if (prob < 0.65) return { label: "High", color: COLORS.high };
  return { label: "Critical", color: COLORS.critical };
}

// --- Segmented control ---
function Segmented({ options, value, onChange }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
      {options.map((opt) => {
        const active = opt === value;
        return (
          <button
            key={opt}
            onClick={() => onChange(opt)}
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              fontSize: 12.5,
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
              border: `1px solid ${active ? COLORS.low : COLORS.border}`,
              background: active ? "rgba(63,174,140,0.14)" : "transparent",
              color: active ? COLORS.low : COLORS.muted,
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
          >
            {opt}
          </button>
        );
      })}
    </div>
  );
}

// --- Slider row ---
function SliderField({ label, value, min, max, step = 1, unit = "", onChange }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 12.5, color: COLORS.muted }}>{label}</span>
        <span
          style={{
            fontSize: 12.5,
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
            color: COLORS.text,
          }}
        >
          {value}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          width: "100%",
          accentColor: COLORS.low,
          height: 4,
          cursor: "pointer",
        }}
      />
    </div>
  );
}

function FieldGroup({ title, children }) {
  return (
    <div
      style={{
        background: COLORS.panel,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 10,
        padding: "16px 18px",
        marginBottom: 14,
      }}
    >
      <div
        style={{
          fontSize: 11,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: COLORS.muted,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          marginBottom: 12,
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}

function ToggleField({ label, value, onChange }) {
  return (
    <div style={{ marginBottom: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: 12.5, color: COLORS.muted }}>{label}</span>
      <Segmented options={["Yes", "No"]} value={value} onChange={onChange} />
    </div>
  );
}

// --- Gauge ---
function Gauge({ probability, color }) {
  const cx = 100, cy = 100, R = 82;
  const zones = [
    { from: 0, to: 0.25, color: COLORS.low },
    { from: 0.25, to: 0.45, color: COLORS.medium },
    { from: 0.45, to: 0.65, color: COLORS.high },
    { from: 0.65, to: 1, color: COLORS.critical },
  ];

  const angleFor = (p) => 180 - 180 * p; // 180deg at p=0, 0deg at p=1
  const pt = (angleDeg, r) => {
    const rad = (angleDeg * Math.PI) / 180;
    return [cx + r * Math.cos(rad), cy - r * Math.sin(rad)];
  };

  const arcPath = (pFrom, pTo, r) => {
    const [x1, y1] = pt(angleFor(pFrom), r);
    const [x2, y2] = pt(angleFor(pTo), r);
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };

  const needleAngle = angleFor(Math.min(Math.max(probability, 0), 1));
  const [nx, ny] = pt(needleAngle, R - 18);

  return (
    <svg viewBox="0 0 200 115" width="100%" height="auto">
      {zones.map((z, i) => (
        <path
          key={i}
          d={arcPath(z.from, z.to, R)}
          fill="none"
          stroke={z.color}
          strokeWidth="14"
          strokeOpacity="0.35"
        />
      ))}
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={color} strokeWidth="3" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="6" fill={color} />
    </svg>
  );
}

export default function ChurnRiskConsole() {
  const [f, setF] = useState({
    age: 42,
    partner: "No",
    dependents: "No",
    tenure_months: 12,
    tech_support: "No",
    online_security: "No",
    streaming_service: "No",
    paperless_billing: "Yes",
    monthly_charges: 70,
    avg_monthly_usage_gb: 90,
    support_calls_6mo: 1,
    late_payments_12mo: 0,
    satisfaction_score: 3,
    contract: "Month-to-month",
    internet_service: "Fiber optic",
    payment_method: "Electronic check",
  });

  const update = (key) => (val) => setF((prev) => ({ ...prev, [key]: val }));

  const { probability, drivers } = useMemo(() => {
    const yn = (v) => (v === "Yes" ? 1 : 0);
    const raw = {
      age: f.age,
      partner: yn(f.partner),
      dependents: yn(f.dependents),
      tenure_months: f.tenure_months,
      tech_support: yn(f.tech_support),
      online_security: yn(f.online_security),
      streaming_service: yn(f.streaming_service),
      paperless_billing: yn(f.paperless_billing),
      monthly_charges: f.monthly_charges,
      avg_monthly_usage_gb: f.avg_monthly_usage_gb,
      support_calls_6mo: f.support_calls_6mo,
      late_payments_12mo: f.late_payments_12mo,
      satisfaction_score: f.satisfaction_score,
      "contract_One year": f.contract === "One year" ? 1 : 0,
      "contract_Two year": f.contract === "Two year" ? 1 : 0,
      "internet_service_Fiber optic": f.internet_service === "Fiber optic" ? 1 : 0,
      "internet_service_No": f.internet_service === "No" ? 1 : 0,
      "payment_method_Credit card": f.payment_method === "Credit card" ? 1 : 0,
      "payment_method_Electronic check": f.payment_method === "Electronic check" ? 1 : 0,
      "payment_method_Mailed check": f.payment_method === "Mailed check" ? 1 : 0,
    };

    let z = MODEL.intercept;
    const contributions = [];
    MODEL.feature_order.forEach((name, i) => {
      const value = raw[name];
      const scaled = (value - MODEL.scaler_mean[i]) / MODEL.scaler_scale[i];
      const contribution = MODEL.coefficients[i] * scaled;
      z += contribution;
      contributions.push({ name, contribution });
    });

    const probability = sigmoid(z);
    const drivers = contributions
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
      .slice(0, 6);

    return { probability, drivers };
  }, [f]);

  const tier = tierFor(probability);
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.contribution)), 0.01);

  const actionText = {
    Low: "Healthy account. No intervention needed \u2014 monitor at next renewal.",
    Medium: "Worth a check-in. Consider a satisfaction survey or loyalty offer.",
    High: "Proactive outreach recommended \u2014 offer a retention incentive or support call.",
    Critical: "Immediate save attempt recommended \u2014 escalate to a retention specialist.",
  }[tier.label];

  return (
    <div
      style={{
        minHeight: "100vh",
        background: COLORS.bg,
        color: COLORS.text,
        fontFamily: "system-ui, -apple-system, sans-serif",
        padding: "28px 20px",
      }}
    >
      <div style={{ maxWidth: 980, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 11,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: COLORS.low,
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
              marginBottom: 6,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <Activity size={13} /> Retention Analytics
          </div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0 }}>Churn Risk Console</h1>
          <p style={{ fontSize: 13, color: COLORS.muted, marginTop: 6, maxWidth: 560 }}>
            Adjust a customer's profile to see live churn probability, powered by a logistic
            regression trained on 5,000 synthetic subscriber records (test AUC {MODEL.test_auc}).
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr",
            gap: 16,
          }}
          className="churn-grid"
        >
          <style>{`
            @media (min-width: 860px) {
              .churn-grid { grid-template-columns: 1fr 360px !important; align-items: start; }
            }
          `}</style>

          {/* LEFT: inputs */}
          <div>
            <FieldGroup title="Customer Profile">
              <SliderField label="Age" value={f.age} min={18} max={85} onChange={update("age")} />
              <ToggleField label="Has partner" value={f.partner} onChange={update("partner")} />
              <ToggleField label="Has dependents" value={f.dependents} onChange={update("dependents")} />
            </FieldGroup>

            <FieldGroup title="Contract & Billing">
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12.5, color: COLORS.muted, marginBottom: 6 }}>Contract type</div>
                <Segmented
                  options={["Month-to-month", "One year", "Two year"]}
                  value={f.contract}
                  onChange={update("contract")}
                />
              </div>
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12.5, color: COLORS.muted, marginBottom: 6 }}>Payment method</div>
                <Segmented
                  options={["Electronic check", "Mailed check", "Bank transfer", "Credit card"]}
                  value={f.payment_method}
                  onChange={update("payment_method")}
                />
              </div>
              <ToggleField label="Paperless billing" value={f.paperless_billing} onChange={update("paperless_billing")} />
              <SliderField
                label="Monthly charges"
                value={f.monthly_charges}
                min={18}
                max={130}
                unit=" USD"
                onChange={update("monthly_charges")}
              />
            </FieldGroup>

            <FieldGroup title="Service">
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12.5, color: COLORS.muted, marginBottom: 6 }}>Internet service</div>
                <Segmented
                  options={["Fiber optic", "DSL", "No"]}
                  value={f.internet_service}
                  onChange={update("internet_service")}
                />
              </div>
              <ToggleField label="Tech support add-on" value={f.tech_support} onChange={update("tech_support")} />
              <ToggleField label="Online security add-on" value={f.online_security} onChange={update("online_security")} />
              <ToggleField label="Streaming add-on" value={f.streaming_service} onChange={update("streaming_service")} />
              <SliderField
                label="Avg. monthly usage"
                value={f.avg_monthly_usage_gb}
                min={0}
                max={500}
                unit=" GB"
                onChange={update("avg_monthly_usage_gb")}
              />
            </FieldGroup>

            <FieldGroup title="Engagement & Support">
              <SliderField label="Tenure" value={f.tenure_months} min={0} max={72} unit=" mo" onChange={update("tenure_months")} />
              <SliderField label="Support calls (6mo)" value={f.support_calls_6mo} min={0} max={10} onChange={update("support_calls_6mo")} />
              <SliderField label="Late payments (12mo)" value={f.late_payments_12mo} min={0} max={10} onChange={update("late_payments_12mo")} />
              <SliderField label="Satisfaction score" value={f.satisfaction_score} min={1} max={5} onChange={update("satisfaction_score")} />
            </FieldGroup>
          </div>

          {/* RIGHT: readout */}
          <div
            style={{
              background: COLORS.panel2,
              border: `1px solid ${COLORS.border}`,
              borderRadius: 12,
              padding: "20px 20px 22px",
              position: "sticky",
              top: 20,
            }}
          >
            <Gauge probability={probability} color={tier.color} />
            <div style={{ textAlign: "center", marginTop: -8 }}>
              <div
                style={{
                  fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                  fontSize: 38,
                  fontWeight: 700,
                  color: tier.color,
                  lineHeight: 1,
                }}
              >
                {(probability * 100).toFixed(1)}%
              </div>
              <div
                style={{
                  fontSize: 12,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  color: tier.color,
                  marginTop: 4,
                  fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                }}
              >
                {tier.label} Risk
              </div>
            </div>

            <p style={{ fontSize: 12.5, color: COLORS.muted, textAlign: "center", marginTop: 12, lineHeight: 1.5 }}>
              {actionText}
            </p>

            <div style={{ height: 1, background: COLORS.border, margin: "16px 0" }} />

            <div
              style={{
                fontSize: 11,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: COLORS.muted,
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                marginBottom: 10,
              }}
            >
              Top Signal Drivers
            </div>

            {drivers.map((d, i) => {
              const pct = (Math.abs(d.contribution) / maxAbs) * 100;
              const increasesRisk = d.contribution > 0;
              return (
                <div key={i} style={{ marginBottom: 9 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                    <span style={{ color: COLORS.text, display: "flex", alignItems: "center", gap: 4 }}>
                      {increasesRisk ? (
                        <ArrowUp size={12} color={COLORS.critical} />
                      ) : (
                        <ArrowDown size={12} color={COLORS.low} />
                      )}
                      {FEATURE_LABELS[d.name] || d.name}
                    </span>
                  </div>
                  <div style={{ height: 5, background: COLORS.border, borderRadius: 3 }}>
                    <div
                      style={{
                        height: "100%",
                        width: `${pct}%`,
                        background: increasesRisk ? COLORS.critical : COLORS.low,
                        borderRadius: 3,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
