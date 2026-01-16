import { useMemo, useState, useEffect, useRef } from "react";
import { Upload, X, Loader2, Image as ImageIcon } from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card } from "@/components/ui/card";
import { useDetectionCanvas } from "@/hooks/useDetectionCanvas";
import { cn } from "@/lib/utils";

type DetectStatus = "idle" | "queued" | "encoding" | "requesting" | "done" | "error";

type ImageJob = {
    id: string;
    file: File;
    previewUrl: string;
    encodedFile?: File;
    encodedPreviewUrl?: string;
    status: DetectStatus;
    detections: any[];
    timing?: any;
    error?: string;
};

export default function ImageDetection() {
    const [jobs, setJobs] = useState<ImageJob[]>([]);
    const [activeId, setActiveId] = useState<string | null>(null);
    const [models, setModels] = useState<string[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>("");
    const [running, setRunning] = useState(false);
    const [maxSideInput, setMaxSideInput] = useState("1280");
    const maxSideRef = useRef(1280);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const activeJob = useMemo(
        () => jobs.find((j) => j.id === activeId) || null,
        [jobs, activeId]
    );

    const { canvasRef } = useDetectionCanvas(
        (activeJob?.encodedPreviewUrl ?? activeJob?.previewUrl) ?? null,
        activeJob?.detections ?? [],
        containerRef
    );

    useEffect(() => {
        async function fetchModels() {
            try {
                const res = await api.getModels();
                const ensembles = res.filter((m: string) => /ensemble/i.test(m));
                setModels(ensembles);
                if (ensembles.length > 0) setSelectedModel(ensembles[0]);
            } catch (e) {
                console.error("Failed to load models");
            }
        }
        fetchModels();
    }, []);

    const addFiles = (files: File[]) => {
        if (files.length === 0) return;
        const newJobs: ImageJob[] = files.map((file) => {
            const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
            return {
                id,
                file,
                previewUrl: URL.createObjectURL(file),
                status: "idle",
                detections: [],
            };
        });

        setJobs((prev) => {
            const merged = [...prev, ...newJobs];
            if (!activeId && merged.length > 0) setActiveId(merged[0].id);
            return merged;
        });
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        addFiles(Array.from(e.target.files || []));
        e.target.value = "";
    };

    const removeJob = (id: string) => {
        setJobs((prev) => {
            const job = prev.find((j) => j.id === id);
            if (job) {
                URL.revokeObjectURL(job.previewUrl);
                if (job.encodedPreviewUrl) URL.revokeObjectURL(job.encodedPreviewUrl);
            }
            const next = prev.filter((j) => j.id !== id);
            if (activeId === id) setActiveId(next[0]?.id ?? null);
            return next;
        });
    };

    const clearAll = () => {
        setJobs((prev) => {
            for (const j of prev) {
                URL.revokeObjectURL(j.previewUrl);
                if (j.encodedPreviewUrl) URL.revokeObjectURL(j.encodedPreviewUrl);
            }
            return [];
        });
        setActiveId(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
    };

    const encodeForDetect = async (srcFile: File): Promise<File> => {
        const maxSide = maxSideRef.current;
        const bitmap = await createImageBitmap(srcFile);
        const needsResize = bitmap.width > maxSide || bitmap.height > maxSide;
        const needsConvert = srcFile.type !== "image/jpeg";

        if (!needsResize && !needsConvert) {
            bitmap.close();
            return srcFile;
        }

        const scale = Math.min(maxSide / bitmap.width, maxSide / bitmap.height, 1);
        const outW = Math.max(1, Math.round(bitmap.width * scale));
        const outH = Math.max(1, Math.round(bitmap.height * scale));

        const canvas = document.createElement("canvas");
        canvas.width = outW;
        canvas.height = outH;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            bitmap.close();
            return srcFile;
        }
        ctx.drawImage(bitmap, 0, 0, outW, outH);
        bitmap.close();

        const blob = await new Promise<Blob | null>((resolve) =>
            canvas.toBlob(resolve, "image/jpeg", 0.9)
        );
        if (!blob) return srcFile;

        const name = srcFile.name.replace(/\.[^.]+$/, "") + ".jpg";
        return new File([blob], name, { type: "image/jpeg" });
    };

    const updateJob = (id: string, patch: Partial<ImageJob>) => {
        setJobs((prev) => prev.map((j) => (j.id === id ? { ...j, ...patch } : j)));
    };

    const mapLimit = async <T,>(
        items: T[],
        limit: number,
        fn: (item: T) => Promise<void>
    ) => {
        const executing = new Set<Promise<void>>();
        for (const item of items) {
            const p = fn(item).finally(() => executing.delete(p));
            executing.add(p);
            if (executing.size >= limit) await Promise.race(executing);
        }
        await Promise.all(executing);
    };

    const handleDetectAll = async () => {
        if (!selectedModel || jobs.length === 0 || running) return;

        const ids = jobs.map((j) => j.id);
        for (const id of ids) updateJob(id, { status: "queued", error: undefined });

        setRunning(true);
        try {
            const concurrency = 4;
            const jobById = new Map(jobs.map((j) => [j.id, j] as const));
            await mapLimit(ids, concurrency, async (id) => {
                const job = jobById.get(id);
                if (!job) return;

                updateJob(id, { status: "encoding" });
                const encoded = await encodeForDetect(job.file);
                if (encoded !== job.file) {
                    const encodedUrl = URL.createObjectURL(encoded);
                    updateJob(id, { encodedFile: encoded, encodedPreviewUrl: encodedUrl });
                } else {
                    updateJob(id, { encodedFile: undefined, encodedPreviewUrl: undefined });
                }

                updateJob(id, { status: "requesting" });
                const res = await api.detectImage(encoded, selectedModel);

                updateJob(id, {
                    status: "done",
                    detections: res.detections || [],
                    timing: res.timing,
                });
            });
        } catch (e: any) {
            console.error(e);
            setJobs((prev) =>
                prev.map((j) =>
                    j.status === "done"
                        ? j
                        : { ...j, status: "error", error: String(e?.message || e) }
                )
            );
        } finally {
            setRunning(false);
        }
    };

    return (
        <div className="flex flex-col h-[calc(100vh-8rem)] gap-6">
            <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold tracking-tight">图像检测</h2>
                <div className="flex items-center gap-4">
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
                    <Select value={selectedModel} onValueChange={setSelectedModel}>
                        <SelectTrigger className="w-[200px]">
                            <SelectValue placeholder="选择模型" />
                        </SelectTrigger>
                        <SelectContent>
                            {models.map((m) => (
                                <SelectItem key={m} value={m}>
                                    {m}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
            </div>

            <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-0">
                <Card className="col-span-2 p-6 flex flex-col items-center justify-center bg-muted/20 border-dashed relative overflow-hidden">
                    <div ref={containerRef} className="relative w-full h-full flex items-center justify-center">
                        {!activeJob ? (
                            <div className="text-center space-y-4">
                                <div className="bg-muted p-4 rounded-full inline-block">
                                    <ImageIcon className="w-10 h-10 text-muted-foreground" />
                                </div>
                                <div className="space-y-1">
                                    <h3 className="font-semibold text-lg">上传图片进行检测</h3>
                                    <p className="text-sm text-muted-foreground">支持批量选择多张图片，并发请求推理</p>
                                </div>
                                <Button variant="outline" className="relative cursor-pointer" asChild>
                                    <label>
                                        <Upload className="mr-2 w-4 h-4" />
                                        添加图片
                                        <input
                                            ref={fileInputRef}
                                            type="file"
                                            className="hidden"
                                            accept="image/*"
                                            multiple
                                            onChange={handleFileChange}
                                        />
                                    </label>
                                </Button>
                            </div>
                        ) : (
                            <>
                                <canvas
                                    ref={canvasRef}
                                    className="max-w-full max-h-full object-contain shadow-lg rounded-md"
                                />
                                <Button
                                    size="icon"
                                    variant="destructive"
                                    className="absolute top-2 right-2 rounded-full shadow-md z-10"
                                    onClick={() => removeJob(activeJob.id)}
                                    disabled={running}
                                >
                                    <X className="w-4 h-4" />
                                </Button>
                            </>
                        )}
                    </div>
                </Card>

                <Card className="flex flex-col p-6 space-y-6">
                    <div className="space-y-2">
                        <h3 className="font-semibold">检测控制</h3>
                        <p className="text-sm text-muted-foreground">支持批量并发检测</p>
                    </div>

                    <div className="flex items-center gap-2">
                        <Button
                            onClick={handleDetectAll}
                            disabled={running || jobs.length === 0 || !selectedModel}
                        >
                            {running ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    检测中…
                                </>
                            ) : (
                                "开始"
                            )}
                        </Button>
                        <Button variant="destructive" onClick={clearAll} disabled={running || jobs.length === 0}>
                            清空
                        </Button>
                    </div>

                    <div className="flex-1 space-y-4 overflow-auto">
                        {jobs.length === 0 ? (
                            <div className="text-center text-sm text-muted-foreground py-10">添加图片后开始检测</div>
                        ) : (
                            <div className="space-y-2">
                                {jobs.map((j) => (
                                    <button
                                        key={j.id}
                                        type="button"
                                        onClick={() => setActiveId(j.id)}
                                        className={cn(
                                            "w-full text-left flex items-center justify-between p-3 rounded-lg text-sm border transition-colors",
                                            j.id === activeId ? "bg-muted" : "bg-muted/50 hover:bg-muted"
                                        )}
                                    >
                                        <div className="min-w-0">
                                            <div className="font-medium truncate">{j.file.name}</div>
                                            <div className="text-xs text-muted-foreground">
                                                {j.status === "idle" && "待检测"}
                                                {j.status === "queued" && "排队中"}
                                                {j.status === "encoding" && "编码中"}
                                                {j.status === "requesting" && "推理中"}
                                                {j.status === "done" && `完成 (${j.detections.length})`}
                                                {j.status === "error" && `失败: ${j.error ?? "unknown"}`}
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            {j.status === "requesting" && (
                                                <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                                            )}
                                            <Button
                                                size="icon"
                                                variant="ghost"
                                                onClick={(ev) => {
                                                    ev.preventDefault();
                                                    ev.stopPropagation();
                                                    removeJob(j.id);
                                                }}
                                                disabled={running}
                                            >
                                                <X className="w-4 h-4" />
                                            </Button>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        )}

                        {activeJob && activeJob.detections.length > 0 && (
                            <div className="space-y-3 animate-in fade-in slide-in-from-right-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium">检测结果 ({activeJob.detections.length})</span>
                                </div>
                                <div className="space-y-2">
                                    {activeJob.detections.map((det, i) => (
                                        <div
                                            key={i}
                                            className="flex items-center justify-between p-3 bg-muted/50 rounded-lg text-sm border hover:bg-muted transition-colors"
                                        >
                                            <div className="flex items-center gap-2">
                                                <div className="w-2 h-2 rounded-full bg-red-500" />
                                                <span className="font-medium capitalize">{det.label}</span>
                                            </div>
                                            <span className="text-muted-foreground">{(det.score * 100).toFixed(1)}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </Card>
            </div>
        </div>
    );
}
