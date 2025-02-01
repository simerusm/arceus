"use client";

import { cn } from "@/lib/utils";
import { Card } from "../../ui/card";
import { useRef, useState, useLayoutEffect, Fragment, useEffect } from "react";
import { useAppContext } from "../../providers/context";
import { Device } from "@/lib/types";

const dimensions = [50, 50, 50, 10];
const displayDimensions = [784, 128, 64, 10];
const initialConnections = generateConnections(dimensions);

export default function NeuralNetworkVisualization({
  pause = false,
  deviceData,
}: {
  pause: boolean;
  deviceData: Device[];
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [layerSpacing, setLayerSpacing] = useState(0);
  const { hoveredDeviceId } = useAppContext();
  const hoveredLayers =
    deviceData.find((d) => d.device_id === hoveredDeviceId)?.device_layers[0] ||
    [];
  const [connections, setConnections] =
    useState<Connection[][]>(initialConnections);

  useLayoutEffect(() => {
    if (!containerRef.current) return;
    const totalWidth = containerRef.current.clientWidth;
    const numGaps = dimensions.length - 1;
    const spacing =
      (totalWidth - dimensions.length * 40) / (dimensions.length + 1);

    setLayerSpacing(spacing);
  }, [dimensions.length]);

  const [animationIndex, setAnimationIndex] = useState(1);
  useEffect(() => {
    if (pause) {
      setAnimationIndex(0);
      return;
    }

    const CYCLE_DURATION = 1000;
    const RESET_PAUSE = 100;

    const interval = setInterval(() => {
      setAnimationIndex((prev) => {
        if (prev === dimensions.length) {
          setTimeout(() => {
            setConnections(generateConnections(dimensions));
            setAnimationIndex(1);
          }, RESET_PAUSE);
          return 0;
        }
        return prev + 1;
      });
    }, CYCLE_DURATION);

    return () => clearInterval(interval);
  }, [dimensions.length, pause]);

  return (
    <Card
      ref={containerRef}
      className="relative z-0 col-span-2 flex items-center justify-evenly font-supply"
    >
      {/* <div className="absolute left-0 top-0">
        {pause ? "PAUSED" : "RUNNING"}
      </div> */}
      <div
        className={cn(
          "absolute -z-20 h-full w-[200%] overflow-visible opacity-25",
          animationIndex === dimensions.length ? "flex" : "hidden",
        )}
      >
        <div
          style={{
            left: `${((layerSpacing + 40) * (dimensions.length - 1)) / 2}px`,
          }}
          className="dotted-pattern ripple-mask absolute h-full w-full bg-primary"
        ></div>
      </div>

      <div
        className="ripple-cover absolute right-0 -z-10 h-full"
        style={{
          width: layerSpacing + 80,
        }}
      ></div>

      {layerSpacing > 0 &&
        dimensions.map((dimension, i) => (
          <Layer
            key={i}
            layer={i + 1}
            numLayers={dimensions.length}
            dimension={dimension}
            spacing={layerSpacing}
            connections={
              i < connections.length
                ? { data: connections[i], nextDimension: dimensions[i + 1] }
                : undefined
            }
            animationIndex={animationIndex}
            hovered={
              hoveredLayers.length === 0 || hoveredLayers.includes(i + 1)
            }
            backpropagating={animationIndex === dimensions.length}
          />
        ))}
    </Card>
  );
}

function Layer({
  layer,
  dimension,
  spacing,
  numLayers,
  animationIndex,
  backpropagating,
  connections,
  hovered,
}: {
  layer: number;
  dimension: number;
  spacing: number;
  numLayers: number;
  animationIndex: number;
  backpropagating: boolean;
  connections?: { data: Connection[]; nextDimension: number };
  hovered: boolean;
}) {
  const lineContainerRef = useRef<HTMLDivElement>(null);
  const [lineContainerHeight, setLineContainerHeight] = useState(0);

  useEffect(() => {
    if (!lineContainerRef.current) return;

    setLineContainerHeight(lineContainerRef.current.clientHeight);

    const observer = new ResizeObserver((entries) => {
      const height = entries[0]?.contentRect.height;
      if (height) setLineContainerHeight(height);
    });

    observer.observe(lineContainerRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      className={cn(
        "flex h-5/6 w-10 flex-col justify-between rounded-lg border bg-nested-card shadow-lg shadow-muted/50",
        hovered ? "opacity-100" : "opacity-25",
      )}
    >
      <div className="h-full py-4">
        <div
          key={
            animationIndex >= layer
              ? "show-visualization"
              : "hide-visualization"
          }
          className={cn(
            "relative flex h-full w-full justify-center",
            animationIndex >= layer ? "opacity-100" : "opacity-0",
          )}
          ref={lineContainerRef}
        >
          {connections &&
            connections.data.map(([from, to], index) => {
              const top = {
                from: from / (dimension - 1),
                to: to / (connections.nextDimension - 1),
              };

              const verticalDistance =
                (top.to - top.from) * lineContainerHeight;
              const angle = Math.atan(verticalDistance / (spacing + 40));
              const lineLength = Math.sqrt(
                (spacing + 40) ** 2 + verticalDistance ** 2,
              );

              return (
                <Fragment key={index}>
                  <div
                    key={backpropagating ? "backprop-start" : "forward-start"}
                    style={{
                      top: `${top.from * 100}%`,
                    }}
                    className={cn(
                      backpropagating ? "nn-backprop-node" : "nn-start-node",
                      "absolute z-10 size-1.5 rounded-full",
                    )}
                  />
                  <div
                    key={backpropagating ? "backprop-line" : "forward-line"}
                    style={{
                      top: `${top.from * 100}%`,
                      width: `${lineLength}px`,
                      transformOrigin: "0 0",
                      transform: `rotate(${angle}rad) translateY(2px)`,
                    }}
                    className={cn(
                      backpropagating
                        ? "nn-backprop-line-pulse"
                        : "nn-line-pulse",
                      "absolute left-5 h-px",
                    )}
                  />
                  <div
                    key={
                      backpropagating && layer < numLayers - 1
                        ? "backprop-end"
                        : "forward-end"
                    }
                    style={{
                      top: `${top.to * 100}%`,
                      transform: `translateX(${spacing + 40}px)`,
                    }}
                    className={cn(
                      backpropagating && layer < numLayers - 1
                        ? "nn-backprop-node"
                        : "nn-end-node",
                      "absolute z-10 size-1.5 rounded-full",
                    )}
                  />
                </Fragment>
              );
            })}
        </div>
      </div>
      <div className="flex flex-col items-center border-t bg-muted/40 py-1 text-sm">
        <div>L{layer}</div>
        <div className="text-muted-foreground">
          {displayDimensions[layer - 1]}
        </div>
      </div>
    </div>
  );
}

type Connection = [number, number];

function generateConnections(dimensions: number[]): Connection[][] {
  const connections: Connection[][] = [];

  for (let layer = 0; layer < dimensions.length - 1; layer++) {
    const fromDim = dimensions[layer];
    const toDim = dimensions[layer + 1];
    const maxConnections = Math.min(fromDim, toDim);

    // Random number of connections between 1 and maxConnections
    const numConnections = Math.max(
      1,
      Math.floor(Math.random() * maxConnections + 1),
    );

    const layerConnections: Connection[] = [];
    for (let i = 0; i < numConnections; i++) {
      const from = Math.floor(Math.random() * fromDim);
      const to = Math.floor(Math.random() * toDim);
      layerConnections.push([from, to]);
    }

    connections.push(layerConnections);
  }

  return connections;
}
