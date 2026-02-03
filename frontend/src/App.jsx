import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { Mic, Check, X, Loader2, Activity, Volume2 } from 'lucide-react'

function App() {
    const [status, setStatus] = useState('idle') // idle, recording, processing, result
    const [timeLeft, setTimeLeft] = useState(8)
    const [prediction, setPrediction] = useState(null)
    const [feedbackSent, setFeedbackSent] = useState(false)

    const mediaRecorderRef = useRef(null)
    const chunksRef = useRef([])
    const timerRef = useRef(null)

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
            mediaRecorderRef.current = new MediaRecorder(stream)
            chunksRef.current = []

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data)
            }

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
                processAudio(blob)
                stream.getTracks().forEach(track => track.stop())
            } // Fixed: Correctly close the brace for onstop

            mediaRecorderRef.current.start()
            setStatus('recording')
            setTimeLeft(8)

            timerRef.current = setInterval(() => {
                setTimeLeft(prev => {
                    if (prev <= 1) {
                        stopRecording() // This is safe because stopRecording clears interval
                        return 0
                    }
                    return prev - 1
                })
            }, 1000)

        } catch (err) {
            console.error("Error accessing microphone:", err)
            alert("Could not access microphone. Please ensure permission is granted.")
        }
    }

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop()
            clearInterval(timerRef.current)
        }
    }

    const processAudio = async (blob) => {
        setStatus('processing')
        const formData = new FormData()
        // Append with filename ending in .webm so backend knows extension
        formData.append('file', blob, 'recording.webm')

        try {
            const response = await axios.post('http://localhost:8000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            })
            setPrediction(response.data)
            setFeedbackSent(false) // Reset feedback status
            setStatus('result')
        } catch (err) {
            console.error("Prediction error:", err)
            setStatus('idle')
            alert("Error processing audio. See console.")
        }
    }

    const handleFeedback = async (isCorrect) => {
        if (!prediction) return

        // If correct, label remains same. If incorrect, flip it (0<->1 for Male/Female)
        // Backend: Female=0, Male=1
        // We assume binary classification here.
        let correctLabel = prediction.label_id
        if (!isCorrect) {
            correctLabel = prediction.label_id === 0 ? 1 : 0
        }

        try {
            await axios.post('http://localhost:8000/feedback', {
                request_id: prediction.request_id,
                correct_label: correctLabel
            })
            setFeedbackSent(true)
            // Optional: Show thank you
        } catch (err) {
            console.error("Feedback error:", err)
        }
    }

    const reset = () => {
        setStatus('idle')
        setPrediction(null)
        setFeedbackSent(false)
        setTimeLeft(8)
    }

    return (
        <div className="container">
            <div className="glass-panel">
                <header style={{ marginBottom: '2rem' }}>
                    <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem', background: 'linear-gradient(to right, #fff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                        Voice Identity AI
                    </h1>
                    <p style={{ color: 'var(--text-secondary)' }}>
                        Speak for 8 seconds to identify gender
                    </p>
                </header>

                {/* Visualizer Placeholder */}
                <div className="visualizer-container">
                    {status === 'recording' || status === 'processing' ? (
                        Array.from({ length: 20 }).map((_, i) => (
                            <div
                                key={i}
                                className="bar"
                                style={{
                                    animationDelay: `${i * 0.05}s`,
                                    height: status === 'processing' ? '20px' : undefined
                                }}
                            />
                        ))
                    ) : (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)' }}>
                            <Activity opacity={0.5} />
                            <span>Ready to record</span>
                        </div>
                    )}
                </div>

                <div style={{ minHeight: '120px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                    {status === 'idle' && (
                        <button className="btn-primary" onClick={startRecording}>
                            <Mic size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                            Start Recording
                        </button>
                    )}

                    {status === 'recording' && (
                        <div style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: '3rem', fontWeight: 'bold', fontFamily: 'monospace' }}>
                                0:0{timeLeft}
                            </div>
                            <p style={{ color: 'var(--accent-secondary)', marginTop: '0.5rem' }}>
                                Recording...
                            </p>
                        </div>
                    )}

                    {status === 'processing' && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <Loader2 className="animate-spin" size={32} color="var(--accent-primary)" />
                            <span style={{ fontSize: '1.2rem' }}>Analyzing voice patterns...</span>
                        </div>
                    )}

                    {status === 'result' && prediction && (
                        <div className="result-card">
                            <div style={{ fontSize: '1rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                                DETECTED GENDER
                            </div>
                            <div style={{
                                fontSize: '3.5rem',
                                fontWeight: '900',
                                background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                marginBottom: '1rem'
                            }}>
                                {prediction.prediction}
                            </div>
                            <div style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>
                                Confidence: {(prediction.confidence * 100).toFixed(1)}%
                            </div>

                            {/* Feedback Section */}
                            <div className="feedback-section">
                                <p style={{ marginBottom: '1rem', fontSize: '0.9rem' }}>
                                    Is this prediction correct?
                                </p>
                                {feedbackSent ? (
                                    <div style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                                        <Check size={20} />
                                        <span>Feedback Saved</span>
                                        <button onClick={reset} style={{ marginLeft: '1rem', background: 'transparent', color: 'var(--text-primary)', textDecoration: 'underline', fontSize: '0.8rem' }}>
                                            Analyze Another
                                        </button>
                                    </div>
                                ) : (
                                    <div style={{ display: 'flex', justifyContent: 'center' }}>
                                        <button
                                            className="feedback-btn active"
                                            onClick={() => handleFeedback(true)}
                                        >
                                            <Check size={18} style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                                            Correct (Default)
                                        </button>
                                        <button
                                            className="feedback-btn"
                                            onClick={() => handleFeedback(false)}
                                        >
                                            <X size={18} style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                                            Incorrect
                                        </button>
                                    </div>
                                )}

                                {!feedbackSent && (
                                    <div style={{ marginTop: '1.5rem' }}>
                                        <button onClick={reset} style={{ background: 'transparent', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                                            Try Again
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default App
