import * as React from "react"
import * as TabsPrimitive from "@radix-ui/react-tabs"

import { cn } from "../../lib/utils"

const Tabs = TabsPrimitive.Root

const TabsList = React.forwardRef(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      "inline-flex h-9 items-center justify-center rounded-lg bg-muted p-1 text-muted-foreground",
      className
    )}
    {...props} />
))
TabsList.displayName = TabsPrimitive.List.displayName

const TabsTrigger = React.forwardRef(({ className, value, children, ...props }, ref) => {
  // Handle empty string values gracefully
  let safeValue = value;
  if (value === "") {
    if (process.env.NODE_ENV !== 'production') {
      console.warn("TabsTrigger: value prop cannot be an empty string. Using fallback value to prevent UI crash.");
    }
    // Generate a fallback value based on children or a random string
    safeValue = typeof children === 'string' ? `tab-${children.replace(/\s+/g, '-').toLowerCase()}` : `tab-fallback-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  return (
    <TabsPrimitive.Trigger
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow",
        className
      )}
      value={safeValue}
      {...props}>
      {children}
    </TabsPrimitive.Trigger>
  );
})
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName

const TabsContent = React.forwardRef(({ className, value, ...props }, ref) => {
  // Handle empty string values gracefully
  let safeValue = value;
  if (value === "") {
    if (process.env.NODE_ENV !== 'production') {
      console.warn("TabsContent: value prop cannot be an empty string. Using fallback value to prevent UI crash.");
    }
    // Generate a fallback value
    safeValue = `content-fallback-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  return (
    <TabsPrimitive.Content
      ref={ref}
      className={cn(
        "mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        className
      )}
      value={safeValue}
      {...props} />
  );
})
TabsContent.displayName = TabsPrimitive.Content.displayName

export { Tabs, TabsList, TabsTrigger, TabsContent }
