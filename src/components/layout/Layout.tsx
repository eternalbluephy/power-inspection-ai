import { Zap, Image as ImageIcon, Video } from "lucide-react";
import { Link, Outlet } from "react-router-dom";
import { NavItem } from "./NavItem";
import { ThemeToggle } from "./ThemeToggle";

export function Layout() {
    return (
        <div className="flex h-screen w-full bg-background transition-colors duration-300">
            {/* Sidebar */}
            <aside className="w-64 border-r border-border/40 bg-card hidden md:flex flex-col">
                <div className="h-16 flex items-center px-6 border-b border-border/40">
                    <Link to="/" className="flex items-center gap-2 font-semibold text-xl text-primary">
                        <div className="bg-primary/10  p-1.5 rounded-lg">
                            <Zap className="h-6 w-6 text-primary fill-primary/20" />
                        </div>
                        <span>电力巡检</span>
                    </Link>
                </div>

                <nav className="flex-1 p-4 space-y-2">
                    <div className="px-4 py-2 mt-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        检测功能
                    </div>
                    <NavItem to="/image" icon={ImageIcon} label="图像检测" />
                    <NavItem to="/video" icon={Video} label="视频检测" />
                </nav>

                <div className="p-4 border-t border-border/40">
                    <div className="text-xs text-center text-muted-foreground">
                        &copy; 2026 Team Spirit
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <div className="flex-1 flex flex-col h-screen overflow-hidden">
                {/* Header */}
                <header className="h-16 border-b border-border/40 bg-background/80 backdrop-blur-md px-6 flex items-center justify-between z-10 sticky top-0">
                    <div className="md:hidden flex items-center gap-2 font-semibold text-lg">
                        <Zap className="h-5 w-5 text-primary" />
                        电力巡检
                    </div>
                    <div className="flex-1" /> {/* Spacer */}
                    <div className="flex items-center gap-4">
                        <ThemeToggle />
                    </div>
                </header>

                {/* Scrollable Content */}
                <main className="flex-1 overflow-auto p-4 md:p-8">
                    <div className="max-w-7xl mx-auto h-full space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <Outlet />
                    </div>
                </main>
            </div>
        </div>
    );
}
