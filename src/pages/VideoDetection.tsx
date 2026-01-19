import { useState, useRef, useEffect } from 'react';
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Upload, Video as VideoIcon, Tv, Camera, Play, Pause } from "lucide-react";
import Webcam from "react-webcam";
import { api } from "@/lib/api";

export default function VideoDetection() {
    const [model, setModel] = useState<string>("");
    const [models, setModels] = useState<string[]>([]);
    const [activeTab, setActiveTab] = useState("upload");
    const [maxSideInput, setMaxSideInput] = useState("640");
    const maxSideRef = useRef(640);
    const [uploadHasVideo, setUploadHasVideo] = useState(false);
    const uploadUrlRef = useRef<string | null>(null);

    const [isCapturing, setIsCapturing] = useState(false);
    const [ws, setWs] = useState<WebSocket | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const isCapturingRef = useRef(false);
    const [detections, setDetections] = useState<any[]>([]);
    const [streamStats, setStreamStats] = useState<{
        fps: number | null;
        frameGapMs: number | null;
        encodeMs: number | null;
        jpegKB: number | null;
        backendDecodeMs: number | null;
        backendLetterboxMs: number | null;
        backendBlobMs: number | null;
        backendTritonMs: number | null;
        backendPostMs: number | null;
        backendTotalMs: number | null;
    }>({
        fps: null,
        frameGapMs: null,
        encodeMs: null,
        jpegKB: null,
        backendDecodeMs: null,
        backendLetterboxMs: null,
        backendBlobMs: null,
        backendTritonMs: null,
        backendPostMs: null,
        backendTotalMs: null,
    });

    const webcamRef = useRef<Webcam>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const sendingRef = useRef(false);
    const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const lastUiUpdateRef = useRef(0);
    const lastStatsUpdateRef = useRef(0);
    const lastFrameTsRef = useRef<number | null>(null);
    const lastEncodeMsRef = useRef<number | null>(null);
    const lastJpegKBRef = useRef<number | null>(null);
    const lastSendStartRef = useRef<number | null>(null);

    const getMaxSendFps = () => 60;
    const scheduleNextCapture = () => {
        const socket = wsRef.current;
        if (!socket || socket.readyState !== WebSocket.OPEN) return;
        if (!isCapturingRef.current) return;

        const maxFps = getMaxSendFps();
        const minIntervalMs = 1000 / Math.max(1, maxFps);
        const lastStart = lastSendStartRef.current;
        const sinceLastStart = lastStart ? performance.now() - lastStart : minIntervalMs;
        const delay = Math.max(0, minIntervalMs - sinceLastStart);

        if (delay <= 0) {
            queueMicrotask(captureFrame);
            return;
        }
        window.setTimeout(captureFrame, delay);
    };

    const [sourceDim, setSourceDim] = useState({ w: 1280, h: 720 });

    useEffect(() => {
        async function fetchModels() {
            try {
                const res = await api.getModels();
                const ensembles = res.filter((m: string) => /ensemble/i.test(m));
                setModels(ensembles);
                if (ensembles.length > 0) setModel(ensembles[0]);
            } catch (e) {
                console.error("Failed to load models");
            }
        }
        fetchModels();
    }, []);

    useEffect(() => {
        return () => stopStream();
    }, []);

    const drawDetections = (data: any) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const { detections, image_size } = data;
        const width = canvas.width;
        const height = canvas.height;
        const baseW = image_size ? image_size[0] : 1280;
        const baseH = image_size ? image_size[1] : 1280;

        detections.forEach((det: any) => {
            const x1 = (det.x1 / baseW) * width;
            const y1 = (det.y1 / baseH) * height;
            const x2 = (det.x2 / baseW) * width;
            const y2 = (det.y2 / baseH) * height;
            const w = x2 - x1;
            const h = y2 - y1;

            ctx.strokeStyle = "#ef4444";
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, w, h);

            ctx.fillStyle = "rgba(239, 68, 68, 0.9)";
            ctx.font = "bold 14px Inter, sans-serif";
            const text = `${det.label} ${(det.score * 100).toFixed(0)}%`;
            const metrics = ctx.measureText(text);
            const textH = 18;
            ctx.fillRect(x1, y1 - textH - 4, metrics.width + 8, textH + 4);

            ctx.fillStyle = "white";
            ctx.fillText(text, x1 + 4, y1 - 6);
        });
    };

    const startStream = () => {
        if (!model) return;
        if (ws && ws.readyState === WebSocket.OPEN) return;
        if (ws) ws.close();

        const url = api.getStreamUrl(model);
        const websocket = new WebSocket(url);
        wsRef.current = websocket;

        websocket.onopen = () => {
            console.log("WS Connected");
            isCapturingRef.current = true;
            setIsCapturing(true);
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            drawDetections(data);

            const now = performance.now();
            if (now - lastUiUpdateRef.current > 200) {
                setDetections(data.detections || []);
                lastUiUpdateRef.current = now;
            }

            const prev = lastFrameTsRef.current;
            lastFrameTsRef.current = now;
            const frameGapMs = prev ? now - prev : null;
            const fps = frameGapMs ? 1000 / Math.max(1, frameGapMs) : null;
            const timing = data?.timing || {};
            const backendDecodeMs = typeof timing.decode_ms === 'number' ? timing.decode_ms : null;
            const backendLetterboxMs = typeof timing.letterbox_ms === 'number' ? timing.letterbox_ms : null;
            const backendBlobMs = typeof timing.blob_ms === 'number' ? timing.blob_ms : null;
            const backendTritonMs = typeof timing.triton_ms === 'number' ? timing.triton_ms : null;
            const backendPostMs = typeof timing.post_ms === 'number' ? timing.post_ms : null;
            const backendTotalMs = typeof timing.total_ms === 'number'
                ? timing.total_ms
                : (typeof data.inference_time === 'number' ? data.inference_time : null);
            if (now - lastStatsUpdateRef.current > 200) {
                setStreamStats({
                    fps,
                    frameGapMs,
                    encodeMs: lastEncodeMsRef.current,
                    jpegKB: lastJpegKBRef.current,
                    backendDecodeMs,
                    backendLetterboxMs,
                    backendBlobMs,
                    backendTritonMs,
                    backendPostMs,
                    backendTotalMs,
                });
                lastStatsUpdateRef.current = now;
            }

            sendingRef.current = false;
            scheduleNextCapture();
        };

        websocket.onerror = (e) => {
            console.error("WS Error", e);
            isCapturingRef.current = false;
            setIsCapturing(false);
            sendingRef.current = false;
        };

        websocket.onclose = () => {
            console.log("WS Closed");
            isCapturingRef.current = false;
            if (wsRef.current === websocket) wsRef.current = null;
            setIsCapturing(false);
            sendingRef.current = false;
        };

        setWs(websocket);
    };

    const stopStream = () => {
        isCapturingRef.current = false;
        setIsCapturing(false);
        sendingRef.current = false;
        if (wsRef.current) wsRef.current.close();
        wsRef.current = null;
        if (ws) ws.close();
        setWs(null);
        if (videoRef.current) {
            videoRef.current.pause();
            if (activeTab === 'screen' && videoRef.current.srcObject) {
                const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
                tracks.forEach(track => track.stop());
                videoRef.current.srcObject = null;
            }
        }
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
    };

    const captureFrame = () => {
        const socket = wsRef.current;
        if (!isCapturingRef.current || !socket || socket.readyState !== WebSocket.OPEN) return;
        if (sendingRef.current) return;

        if (!captureCanvasRef.current) {
            captureCanvasRef.current = document.createElement('canvas');
        }
        const tempCanvas = captureCanvasRef.current;

        let videoSource: HTMLVideoElement | null = null;
        if (activeTab === 'webcam' && webcamRef.current) {
            videoSource = webcamRef.current.video;
        } else if ((activeTab === 'upload' || activeTab === 'screen') && videoRef.current) {
            videoSource = videoRef.current;
        }

        if (videoSource) {
            const vw = videoSource.videoWidth;
            const vh = videoSource.videoHeight;
            if (vw === 0 || vh === 0) return;
            if (vw > 0 && vh > 0 && (vw !== sourceDim.w || vh !== sourceDim.h)) {
                setSourceDim({ w: vw, h: vh });
            }

            const maxSide = maxSideRef.current;
            const scale = Math.min(maxSide / vw, maxSide / vh, 1);
            tempCanvas.width = Math.round(vw * scale);
            tempCanvas.height = Math.round(vh * scale);

            if (videoSource.readyState >= 2) {
                const ctx = tempCanvas.getContext('2d');
                if (ctx) {
                    ctx.drawImage(videoSource, 0, 0, tempCanvas.width, tempCanvas.height);
                    sendingRef.current = true;
                    lastSendStartRef.current = performance.now();
                    const encodeStart = performance.now();
                    tempCanvas.toBlob((blob) => {
                        try {
                            if (!blob) {
                                sendingRef.current = false;
                                scheduleNextCapture();
                                return;
                            }
                            lastEncodeMsRef.current = performance.now() - encodeStart;
                            lastJpegKBRef.current = blob.size / 1024;
                            socket.send(blob);
                        } catch (e) {
                            console.error("WS send failed", e);
                            sendingRef.current = false;
                        }
                    }, 'image/jpeg', activeTab === 'screen' ? 0.55 : 0.7);
                }
            }
        }
    };

    useEffect(() => {
        if (!isCapturing) return;
        scheduleNextCapture();
    }, [isCapturing, ws, activeTab]);

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file && videoRef.current) {
            const url = URL.createObjectURL(file);
            if (uploadUrlRef.current) URL.revokeObjectURL(uploadUrlRef.current);
            uploadUrlRef.current = url;
            videoRef.current.src = url;
            videoRef.current.load();
            setUploadHasVideo(true);
            stopStream();
        }
    };

    const handleScreenShare = async () => {
        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
                startStream();
            }
        } catch (e) {
            console.error("Screen share failed", e);
        }
    };

    const handleTabChange = (val: string) => {
        stopStream();
        setActiveTab(val);
        if (videoRef.current) {
            videoRef.current.pause();
            videoRef.current.removeAttribute("src");
            videoRef.current.load();
            videoRef.current.srcObject = null;
        }
        if (uploadUrlRef.current) URL.revokeObjectURL(uploadUrlRef.current);
        uploadUrlRef.current = null;
        setUploadHasVideo(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
        setWs(null);
    };

    const onVideoMetadata = () => {
        if (videoRef.current) {
            setSourceDim({ w: videoRef.current.videoWidth, h: videoRef.current.videoHeight });
        }
    };

    const onWebcamUserMedia = (stream: MediaStream) => {
        const track = stream.getVideoTracks()[0];
        const settings = track.getSettings();
        if (settings.width && settings.height) {
            setSourceDim({ w: settings.width, h: settings.height });
        }
    };

    return (
        <div className="flex flex-col lg:h-[calc(100vh-8rem)] gap-6">
            <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold tracking-tight">视频检测</h2>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2">
                        <Label htmlFor="maxSide" className="text-xs text-muted-foreground">maxSide</Label>
                        <Input
                            id="maxSide"
                            className="w-[110px]"
                            type="number"
                            inputMode="numeric"
                            min={160}
                            max={1920}
                            step={10}
                            value={maxSideInput}
                            onChange={(e) => {
                                const v = e.target.value;
                                setMaxSideInput(v);
                                const n = Number.parseInt(v, 10);
                                if (!Number.isFinite(n)) return;
                                maxSideRef.current = Math.min(1920, Math.max(160, n));
                            }}
                        />
                    </div>

                    <Select value={model} onValueChange={setModel}>
                        <SelectTrigger className="w-[200px]">
                            <SelectValue placeholder="选择模型" />
                        </SelectTrigger>
                        <SelectContent>
                            {models.map((m) => (
                                <SelectItem key={m} value={m}>{m}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
            </div>

            <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 min-h-0">
                <Card className="col-span-3 p-6 relative overflow-hidden bg-muted/20 flex flex-col">
                    <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full flex-1 flex flex-col">
                        <TabsList className="grid w-full grid-cols-3 mb-4">
                            <TabsTrigger value="upload">
                                <VideoIcon className="mr-2 w-4 h-4" /> 文件上传
                            </TabsTrigger>
                            <TabsTrigger value="webcam">
                                <Camera className="mr-2 w-4 h-4" /> 摄像头
                            </TabsTrigger>
                            <TabsTrigger value="screen">
                                <Tv className="mr-2 w-4 h-4" /> 屏幕共享
                            </TabsTrigger>
                        </TabsList>

                        <div className="flex-1 min-h-[400px] lg:min-h-0 relative bg-black/90 rounded-lg overflow-hidden flex items-center justify-center border border-border/50 shadow-inner group">

                            <canvas
                                ref={canvasRef}
                                className="absolute inset-0 z-20 pointer-events-none w-full h-full object-contain"
                                width={sourceDim.w}
                                height={sourceDim.h}
                            />

                            <TabsContent value="upload" className="w-full h-full mt-0 flex flex-col items-center justify-center relative">
                                <input
                                    type="file"
                                    ref={fileInputRef}
                                    className="hidden"
                                    accept="video/*"
                                    onChange={handleFileUpload}
                                />
                                <video
                                    ref={videoRef}
                                    className="absolute inset-0 w-full h-full object-contain z-10"
                                    playsInline
                                    muted
                                    loop
                                    onLoadedMetadata={onVideoMetadata}
                                    onPlay={() => !isCapturing && startStream()}
                                    onPause={() => isCapturing && stopStream()}
                                />

                                {!uploadHasVideo && (
                                    <div className="z-30 flex flex-col items-center gap-4">
                                        <div className="p-4 bg-muted/10 rounded-full backdrop-blur-sm border border-white/10">
                                            <VideoIcon className="w-12 h-12 text-muted-foreground" />
                                        </div>
                                        <Button variant="secondary" onClick={() => fileInputRef.current?.click()}>
                                            <Upload className="mr-2 w-4 h-4" />
                                            上传视频文件
                                        </Button>
                                    </div>
                                )}

                                {uploadHasVideo && (
                                    <div className="absolute bottom-6 z-30 flex gap-4 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <Button
                                            variant="secondary"
                                            size="icon"
                                            onClick={() => {
                                                if (videoRef.current?.paused) videoRef.current.play();
                                                else videoRef.current?.pause();
                                            }}
                                        >
                                            {isCapturing ? <Pause className="fill-current" /> : <Play className="fill-current" />}
                                        </Button>
                                        <Button
                                            variant="destructive"
                                            onClick={() => {
                                                stopStream();
                                                if (videoRef.current) {
                                                    videoRef.current.pause();
                                                    videoRef.current.removeAttribute("src");
                                                    videoRef.current.load();
                                                }
                                                if (uploadUrlRef.current) URL.revokeObjectURL(uploadUrlRef.current);
                                                uploadUrlRef.current = null;
                                                setUploadHasVideo(false);
                                                setDetections([]);
                                                if (fileInputRef.current) fileInputRef.current.value = "";
                                            }}
                                        >
                                            清除
                                        </Button>
                                    </div>
                                )}
                            </TabsContent>

                            <TabsContent value="webcam" className="w-full h-full mt-0 relative">
                                <Webcam
                                    ref={webcamRef}
                                    audio={false}
                                    screenshotFormat="image/jpeg"
                                    className="w-full h-full object-contain"
                                    videoConstraints={{ width: 1280, height: 720 }}
                                    onUserMedia={onWebcamUserMedia}
                                />
                                <div className="absolute bottom-6 left-0 right-0 flex justify-center z-30 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <Button
                                        variant={isCapturing ? "destructive" : "default"}
                                        onClick={isCapturing ? stopStream : startStream}
                                    >
                                        {isCapturing ? "停止检测" : "开启摄像头检测"}
                                    </Button>
                                </div>
                            </TabsContent>

                            <TabsContent value="screen" className="w-full h-full mt-0 relative">
                                <video
                                    ref={videoRef}
                                    className="w-full h-full object-contain"
                                    autoPlay
                                    muted
                                    playsInline
                                    onLoadedMetadata={onVideoMetadata}
                                />
                                <div className="absolute bottom-6 left-0 right-0 flex justify-center z-30 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <Button
                                        variant={isCapturing ? "destructive" : "default"}
                                        onClick={isCapturing ? stopStream : handleScreenShare}
                                    >
                                        {isCapturing ? "停止共享" : "开始屏幕共享"}
                                    </Button>
                                </div>
                            </TabsContent>
                        </div>
                    </Tabs>
                </Card>

                <Card className="col-span-1 p-6 flex flex-col space-y-4 lg:h-full h-[300px] overflow-hidden">
                    <div className="space-y-2">
                        <h3 className="font-semibold">实时检测结果</h3>
                        <p className="text-sm text-muted-foreground">{isCapturing ? "正在分析视频流..." : "等待视频源"}</p>
                        {isCapturing && (
                            <div className="text-xs text-muted-foreground grid grid-cols-2 gap-x-3 gap-y-1">
                                <span>接收帧率：{streamStats.fps !== null ? `${streamStats.fps.toFixed(1)}fps` : "-"}</span>
                                <span>帧间隔：{streamStats.frameGapMs !== null ? `${streamStats.frameGapMs.toFixed(1)}ms` : "-"}</span>
                                <span>编码耗时：{streamStats.encodeMs !== null ? `${streamStats.encodeMs.toFixed(1)}ms` : "-"}</span>
                                <span>JPEG大小：{streamStats.jpegKB !== null ? `${streamStats.jpegKB.toFixed(0)}KB` : "-"}</span>
                                <span>后端总耗时：{streamStats.backendTotalMs !== null ? `${streamStats.backendTotalMs.toFixed(1)}ms` : "-"}</span>
                                <span>Triton耗时：{streamStats.backendTritonMs !== null ? `${streamStats.backendTritonMs.toFixed(1)}ms` : "-"}</span>
                                <span>后端解码：{streamStats.backendDecodeMs !== null ? `${streamStats.backendDecodeMs.toFixed(1)}ms` : "-"}</span>
                                <span>后处理：{streamStats.backendPostMs !== null ? `${streamStats.backendPostMs.toFixed(1)}ms` : "-"}</span>
                                <span>letterbox：{streamStats.backendLetterboxMs !== null ? `${streamStats.backendLetterboxMs.toFixed(1)}ms` : "-"}</span>
                                <span>blob：{streamStats.backendBlobMs !== null ? `${streamStats.backendBlobMs.toFixed(1)}ms` : "-"}</span>
                            </div>
                        )}
                    </div>

                    <div className="flex-1 overflow-auto space-y-2 pr-2 custom-scrollbar">
                        {detections.length === 0 && isCapturing && (
                            <div className="text-center text-sm text-muted-foreground py-10">暂无异常发现</div>
                        )}
                        {detections.map((det, i) => (
                            <div key={i} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg text-sm border hover:bg-muted transition-colors animate-in fade-in slide-in-from-right-2 duration-300">
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-red-500" />
                                    <span className="font-medium capitalize">{det.label}</span>
                                </div>
                                <span className="text-muted-foreground">{(det.score * 100).toFixed(0)}%</span>
                            </div>
                        ))}
                    </div>
                </Card>
            </div>
        </div>
    );
}
