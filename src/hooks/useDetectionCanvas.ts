import { useEffect, useRef, useState } from "react";

interface Detection {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    score: number;
    label: string;
}

export function useDetectionCanvas(
    imageSrc: string | null,
    detections: Detection[],
    containerRef: React.RefObject<HTMLDivElement | null>
) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
    const renderSeqRef = useRef(0);

    useEffect(() => {
        if (!imageSrc) return;
        const img = new Image();
        img.src = imageSrc;
        img.onload = () => {
            setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
        };
    }, [imageSrc]);

    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container || !imageSrc || !naturalSize) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const { offsetWidth, offsetHeight } = container;
        if (offsetWidth <= 0 || offsetHeight <= 0) return;

        const scale = Math.min(offsetWidth / naturalSize.w, offsetHeight / naturalSize.h);
        const displayWidth = Math.max(1, Math.round(naturalSize.w * scale));
        const displayHeight = Math.max(1, Math.round(naturalSize.h * scale));

        const dpr = window.devicePixelRatio || 1;
        canvas.style.width = `${displayWidth}px`;
        canvas.style.height = `${displayHeight}px`;
        canvas.width = Math.max(1, Math.round(displayWidth * dpr));
        canvas.height = Math.max(1, Math.round(displayHeight * dpr));

        const renderSeq = ++renderSeqRef.current;

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const img = new Image();
        img.src = imageSrc;

        const draw = () => {
            if (renderSeq !== renderSeqRef.current) return;

            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, displayWidth, displayHeight);

            ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

            const sx = displayWidth / naturalSize.w;
            const sy = displayHeight / naturalSize.h;

            detections.forEach((det) => {
                const { x1, y1, x2, y2, score, label } = det;
                const rx1 = x1 * sx;
                const ry1 = y1 * sy;
                const width = (x2 - x1) * sx;
                const height = (y2 - y1) * sy;

                ctx.strokeStyle = "#ef4444"; // red-500
                ctx.lineWidth = 3;
                ctx.strokeRect(rx1, ry1, width, height);

                const text = `${label} ${(score * 100).toFixed(1)}%`;
                ctx.font = "bold 16px Inter, sans-serif";
                const textMetrics = ctx.measureText(text);
                const textHeight = 16;
                const padding = 6;
                const bgW = textMetrics.width + padding * 2;
                const bgH = textHeight + padding * 2;
                const bgX = rx1;
                const bgY = Math.max(0, ry1 - bgH);

                ctx.fillStyle = "rgba(239, 68, 68, 0.85)";
                ctx.fillRect(bgX, bgY, bgW, bgH);

                ctx.fillStyle = "#ffffff";
                ctx.fillText(text, bgX + padding, bgY + textHeight + (padding - 2));
            });
        };

        img.onload = () => draw();
        if (img.complete && img.naturalWidth > 0) draw();

    }, [imageSrc, detections, naturalSize, containerRef]);

    return { canvasRef };
}
