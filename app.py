from flask import Flask, request, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load selected features
with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

# Load numerical selected
with open('numerical_selected.pkl', 'rb') as f:
    numerical_selected = pickle.load(f)

html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Student Performance Predictor</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            color-scheme: dark;
            --bg-1: #08101f;
            --bg-2: #0e1a32;
            --surface: rgba(15, 23, 42, 0.76);
            --surface-strong: rgba(15, 23, 42, 0.92);
            --border: rgba(255, 255, 255, 0.12);
            --text: #e2e8f0;
            --muted: #94a3b8;
            --accent: #7c3aed;
            --success: #22c55e;
            --danger: #ef4444;
            --shadow: 0 32px 88px rgba(8, 15, 39, 0.42);
        }

        * {
            box-sizing: border-box;
        }

        html, body {
            margin: 0;
            min-height: 100%;
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: var(--text);
        }

        body {
            background: radial-gradient(circle at top left, rgba(124, 58, 237, 0.18), transparent 28%),
                        radial-gradient(circle at bottom right, rgba(34, 197, 94, 0.14), transparent 32%),
                        linear-gradient(180deg, var(--bg-1), var(--bg-2));
            display: grid;
            place-items: center;
            padding: 32px;
            min-height: 100vh;
            overflow-x: hidden;
        }

        body::after {
            content: '';
            position: fixed;
            inset: 0;
            background: radial-gradient(circle at 16% 12%, rgba(124, 58, 237, 0.22), transparent 24%),
                        radial-gradient(circle at 84% 82%, rgba(34, 197, 94, 0.16), transparent 20%);
            z-index: -1;
            pointer-events: none;
        }

        .app-shell {
            width: min(100%, 1160px);
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.05fr 0.95fr;
            gap: 28px;
            align-items: stretch;
        }

        .panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 32px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(24px);
            overflow: hidden;
            position: relative;
            animation: fadeInUp 0.9s ease both;
        }

        .floating {
            animation: floatCard 8s ease-in-out infinite alternate, fadeInUp 0.9s ease both;
        }

        .panel::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.12), transparent 30%);
            pointer-events: none;
        }

        .hero-panel {
            padding: 42px 40px 36px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 26px;
        }

        .hero-heading {
            margin: 0;
            font-size: clamp(3rem, 4vw, 4.6rem);
            line-height: 1.02;
            font-weight: 800;
            letter-spacing: -0.05em;
            color: #f8fafc;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            color: #a5b4fc;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.78rem;
            margin-bottom: 8px;
        }

        .title {
            margin: 0;
            font-size: clamp(1.5rem, 2.6vw, 2.4rem);
            line-height: 1.15;
            letter-spacing: -0.03em;
            color: var(--text);
        }

        .subtitle {
            margin: 0;
            color: var(--muted);
            line-height: 1.8;
            max-width: 560px;
            font-size: 1rem;
        }

        .tag-list {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }

        .tag {
            padding: 10px 16px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.06);
            color: var(--text);
            font-size: 0.92rem;
        }

        .card {
            padding: 36px;
            display: grid;
            gap: 24px;
            min-height: 100%;
        }

        .section-title {
            margin: 0;
            font-size: 1.35rem;
            font-weight: 700;
        }

        .input-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 18px;
        }

        .field {
            display: grid;
            gap: 10px;
        }

        .field label {
            display: flex;
            align-items: center;
            gap: 12px;
            color: var(--muted);
            font-size: 0.96rem;
            font-weight: 600;
        }

        .field .icon {
            width: 22px;
            height: 22px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: rgba(124, 58, 237, 0.14);
            color: #c7d2fe;
            font-size: 0.95rem;
        }

        .field input,
        .field select {
            width: 100%;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.62);
            color: var(--text);
            padding: 16px 18px;
            font-size: 1rem;
            transition: border-color 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease;
        }

        .field input::placeholder {
            color: rgba(226, 232, 240, 0.45);
        }

        .field input:focus,
        .field select:focus {
            outline: none;
            border-color: rgba(99, 102, 241, 0.78);
            box-shadow: 0 0 0 6px rgba(99, 102, 241, 0.12);
            transform: translateY(-1px);
        }

        .form-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 8px;
        }

        .cta-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 240px;
            padding: 18px 24px;
            border: none;
            border-radius: 18px;
            background: linear-gradient(135deg, #7c3aed 0%, #22d3ee 100%);
            color: white;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            cursor: pointer;
            box-shadow: 0 20px 42px rgba(124, 58, 237, 0.3);
            transition: transform 0.25s ease, box-shadow 0.25s ease, opacity 0.25s ease;
            position: relative;
            overflow: hidden;
        }

        .cta-btn:hover {
            transform: translateY(-2px);
        }

        .cta-btn:disabled {
            opacity: 0.7;
            cursor: wait;
            box-shadow: none;
        }

        .cta-btn::before {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at top left, rgba(255,255,255,0.32), transparent 25%);
            opacity: 0.4;
            pointer-events: none;
        }

        .spinner {
            display: none;
            width: 18px;
            height: 18px;
            border: 3px solid rgba(255,255,255,0.35);
            border-top-color: white;
            border-radius: 50%;
            margin-left: 12px;
            animation: spin 0.9s linear infinite;
        }

        .result-card {
            border-radius: 28px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.06);
            box-shadow: 0 28px 70px rgba(8, 15, 39, 0.2);
            opacity: 0;
            transform: translateY(18px);
            transition: opacity 0.45s ease, transform 0.45s ease;
        }

        .result-card.visible {
            opacity: 1;
            transform: translateY(0);
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 22px;
        }

        .result-label {
            color: var(--muted);
            font-size: 0.95rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .result-badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 12px 18px;
            border-radius: 999px;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.86rem;
            border: 1px solid transparent;
        }

        .result-badge.good {
            color: #86efac;
            background: rgba(34, 197, 94, 0.14);
            border-color: rgba(34, 197, 94, 0.25);
        }

        .result-badge.poor {
            color: #fecaca;
            background: rgba(239, 68, 68, 0.14);
            border-color: rgba(239, 68, 68, 0.25);
        }

        .result-value {
            font-size: clamp(3rem, 5vw, 4.2rem);
            line-height: 1;
            margin: 0;
            letter-spacing: -0.05em;
        }

        .result-copy {
            margin: 16px 0 0;
            color: #cbd5e1;
            line-height: 1.75;
            font-size: 1rem;
            max-width: 100%;
        }

        .result-summary {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            margin-top: 22px;
        }

        .summary-pill {
            border-radius: 16px;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.04);
            color: #e2e8f0;
            font-size: 0.95rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes floatCard {
            0% { transform: translateY(0px); }
            100% { transform: translateY(-10px); }
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 980px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 680px) {
            body {
                padding: 18px;
            }
            .hero-panel {
                padding: 32px 24px 28px;
            }
            .card {
                padding: 28px 22px;
            }
            .input-grid {
                grid-template-columns: 1fr;
            }
            .form-footer {
                flex-direction: column;
                align-items: stretch;
            }
            .cta-btn {
                min-width: 100%;
            }
            .result-header {
                flex-direction: column;
                align-items: flex-start;
            }
        }
    </style>
</head>
<body>
    <div class="app-shell">
        <div class="hero-grid">
            <section class="panel floating hero-panel">
                <h1 class="hero-heading">Student Performance Prediction</h1>
                <p class="title">Transforming student results into intelligent outcomes</p>
            </section>

            <section class="panel card">
                <h2 class="section-title">Model Inputs</h2>
                <form id="predictionForm" method="POST" action="/predict">
                    <div class="input-grid">
                        <div class="field">
                            <label for="age"><span class="icon">👤</span>Age</label>
                            <input type="number" id="age" name="age" placeholder="18" required min="14" max="25">
                        </div>
                        <div class="field">
                            <label for="study_hours"><span class="icon">📘</span>Study Hours</label>
                            <input type="number" id="study_hours" name="study_hours" placeholder="4.5" required step="0.1" min="0" max="12">
                        </div>
                        <div class="field">
                            <label for="attendance_percentage"><span class="icon">📅</span>Attendance %</label>
                            <input type="number" id="attendance_percentage" name="attendance_percentage" placeholder="92" required min="0" max="100">
                        </div>
                        <div class="field">
                            <label for="computer_network_score"><span class="icon">🌐</span>Computer Network Score</label>
                            <input type="number" id="computer_network_score" name="computer_network_score" placeholder="82" required min="0" max="100">
                        </div>
                        <div class="field">
                            <label for="operating_system_score"><span class="icon">🖥️</span>Operating System Score</label>
                            <input type="number" id="operating_system_score" name="operating_system_score" placeholder="78" required min="0" max="100">
                        </div>
                        <div class="field">
                            <label for="daa_score"><span class="icon">⚙️</span>DAA Score</label>
                            <input type="number" id="daa_score" name="daa_score" placeholder="85" required min="0" max="100">
                        </div>
                        <div class="field">
                            <label for="overall_score"><span class="icon">📊</span>Overall Score</label>
                            <input type="number" id="overall_score" name="overall_score" placeholder="80" readonly min="0" max="100">
                        </div>
                        <div class="field">
                            <label for="gender"><span class="icon">⚧</span>Gender</label>
                            <select id="gender" name="gender">
                                <option value="male">Male</option>
                                <option value="female">Female</option>
                                <option value="other">Other</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="school_type"><span class="icon">🏫</span>School Type</label>
                            <select id="school_type" name="school_type">
                                <option value="public">Public</option>
                                <option value="private">Private</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="internet_access"><span class="icon">📶</span>Internet Access</label>
                            <select id="internet_access" name="internet_access">
                                <option value="yes">Yes</option>
                                <option value="no">No</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-footer">
                        <button class="cta-btn" type="submit">
                            <span>Predict Final Grade</span>
                            <div class="spinner" id="formSpinner"></div>
                        </button>
                    </div>
                </form>

                {% if result %}
                <div class="result-card visible">
                    <div class="result-header">
                        <div>
                            <div class="result-label">Prediction Result</div>
                            <p class="result-copy">The AI model has generated the final grade estimate for the provided student profile.</p>
                        </div>
                        <span class="result-badge {{ result_class }}">{{ result_class | capitalize }}</span>
                    </div>
                    <div class="result-value">{{ result }}</div>
                    <p class="result-copy">{{ result_message }}</p>
                    {% if grade_recommendation %}
                    <div class="result-summary">
                        <p class="result-copy" style="margin: 0; font-size: 1.05rem; font-weight: 600; color: #fbbf24;">{{ grade_recommendation }}</p>
                    </div>
                    {% endif %}
                </div>
                {% endif %}
            </section>
        </div>
    </div>

    <script>
        const form = document.getElementById('predictionForm');
        const button = document.querySelector('.cta-btn');
        const spinner = document.getElementById('formSpinner');
        const overallScore = document.getElementById('overall_score');
        const subjectFields = [
            document.getElementById('computer_network_score'),
            document.getElementById('operating_system_score'),
            document.getElementById('daa_score')
        ];

        function updateOverallScore() {
            const values = subjectFields.map(field => parseFloat(field.value));
            if (values.some(value => Number.isNaN(value))) {
                overallScore.value = '';
                return;
            }
            const average = values.reduce((sum, value) => sum + value, 0) / values.length;
            overallScore.value = Math.round(average * 10) / 10;
        }

        subjectFields.forEach(field => {
            field.addEventListener('input', updateOverallScore);
        });

        form.addEventListener('submit', () => {
            button.disabled = true;
            spinner.style.display = 'inline-block';
            updateOverallScore();
        });
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def home():
    return render_template_string(html_template, result=None, result_class='good', result_message='', grade_recommendation='')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {}
    for key in request.form:
        if key in ['age', 'study_hours', 'attendance_percentage', 'computer_network_score', 'operating_system_score', 'daa_score', 'overall_score']:
            data[key] = float(request.form[key])
        else:
            data[key] = request.form[key]
    
    # Create DataFrame
    input_df = pd.DataFrame([data])
    
    # One-hot encode the input
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    
    # Ensure all selected_features are present, fill missing with 0
    for feature in selected_features:
        if feature not in input_encoded.columns:
            input_encoded[feature] = 0
    
    # Reorder to match selected_features
    input_encoded = input_encoded[selected_features]
    
    # Scale numerical features
    input_encoded_copy = input_encoded.copy()
    input_encoded_copy[numerical_selected] = scaler.transform(input_encoded[numerical_selected])
    
    # Predict
    prediction = model.predict(input_encoded_copy)
    prediction_label = prediction[0]
    result_raw = str(prediction_label).strip()
    result = result_raw.upper() if result_raw.isalpha() else result_raw

    grade_messages = {
        "A": "Outstanding performance, keep excelling ahead",
        "B": "Great work, improve consistency slightly",
        "C": "Average performance, needs more focus",
        "D": "Below average, increase effort and practice",
        "E": "Poor performance, immediate improvement required"
    }

    if result in grade_messages:
        result_class = 'good' if result in ['A', 'B'] else 'poor'
        result_message = grade_messages[result]
    else:
        result_class = 'good'
        result_message = 'Prediction generated successfully.'

    # Generate grade recommendation for non-A grades
    grade_recommendation = ''
    if result != 'A':
        # Calculate required overall score for grade A
        # Estimate: typically A grade requires overall score of 85+
        current_overall = float(data.get('overall_score', 0))
        required_score = 85
        improvement_needed = max(0, required_score - current_overall)
        
        if improvement_needed > 0:
            grade_recommendation = f"📈 To achieve Grade A, you need an overall score of {required_score}. You need to improve by {improvement_needed:.1f} marks."
        else:
            grade_recommendation = f"🎯 You're close! Maintain an overall score of {required_score}+ to secure Grade A."
    else:
        grade_recommendation = "🏆 Congratulations! You have achieved the highest grade (A). Keep up the excellent performance!"

    return render_template_string(html_template, result=result, result_class=result_class, result_message=result_message, grade_recommendation=grade_recommendation)

if __name__ == '__main__':
    app.run(debug=True)