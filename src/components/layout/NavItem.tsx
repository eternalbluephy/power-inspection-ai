import { type LucideIcon } from "lucide-react";
import { useLocation, Link } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface NavItemProps {
    to: string;
    icon: LucideIcon;
    label: string;
}

export function NavItem({ to, icon: Icon, label }: NavItemProps) {
    const location = useLocation();
    const isActive = location.pathname === to;

    return (
        <Link to={to}>
            <Button
                variant="ghost"
                className={cn(
                    "w-full justify-start gap-3 h-12 text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors",
                    isActive && "bg-muted text-foreground font-medium"
                )}
            >
                <Icon className="h-5 w-5" />
                {label}
            </Button>
        </Link>
    );
}
