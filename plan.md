# Development Plan — LLM Neuroscope (MVP)

This document serves as the roadmap for implementing the design spec. Each task should be checked off as it is completed.

## Phase 1: Foundations & Design System
- [ ] Configure the color palette (CSS / Tailwind tokens): dark backgrounds, blue accent, semantic colors.
- [ ] Configure typography (Inter/Roboto) with defined sizes and weights.
- [ ] Create base UI components:
  - [ ] Buttons (Primary, Secondary, Danger).
  - [ ] Status badges (Success, Warning, Danger).
  - [ ] Inputs and Dropdowns.
  - [ ] Cards with borders and border-radius.

## Phase 2: Global Layout & Navigation
- [ ] Develop the Sidebar (left):
  - [ ] Import zone (Drag & Drop + "Choose a file" button).
  - [ ] List of loaded datasets with metadata.
- [ ] Develop the Header (top):
  - [ ] App title and subtitle.
  - [ ] Active dataset indicator and counter (e.g., 240 records · 3 runs).
  - [ ] Navigation bar (Tabs): Overview, Activity, Anomalies, Comparison, Neuron Probe, Evals.
  - [ ] Model selector (Dropdown).
  - [ ] Export buttons (CSV, PDF).
- [ ] Implement routing system between tabs.

## Phase 3: Screen Development (Views)

### 3.1 Tab: Overview
- [ ] Integrate the 6 KPI cards (Perplexity, Token entropy, Degeneracy rate, Max drift, Avg length, Avg latency).
- [ ] Integrate the Bar chart (Entropy per run) with displayed values.
- [ ] Integrate the Line chart (Avg log-prob & entropy per position) with a clickable legend.

### 3.2 Tab: Activity
- [ ] Integrate the Activity Heatmap (Entropy per position x run) with color gradient.
- [ ] Add the Heatmap horizontal legend.
- [ ] Integrate the histogram (Entropy profile per position).

### 3.3 Tab: Anomalies
- [ ] Integrate the 4 KPI cards (Detected anomalies, Repetition loop, Latency spike, Blocking points).
- [ ] Develop Table 1 (Anomaly log) with severity badges.
- [ ] Develop Table 2 (Blocking points per position).
- [ ] *UX*: Add a "View prompt" action on log rows.

### 3.4 Tab: Comparison
- [ ] Integrate the header with the two model selectors (A/B).
- [ ] Add the global status badge (e.g., "IMPROVEMENT") and the divergence metric.
- [ ] Develop the comparison table (Metric, Model A, Model B, Delta, Delta %, Status).
- [ ] Implement column sorting in the table.

### 3.5 Tab: Neuron Probe
- [ ] *UX*: Create an accordion or sub-tabs system to manage page density.
- [ ] Integrate the "Instrumented Generation" zone (Input, button, Output, Perplexity).
- [ ] Integrate the "Activation analysis per layer" (Progress bars, badges).
- [ ] Integrate the "Attention entropy" Heatmap (Layer x Head) with tooltips.
- [ ] Integrate the "Token attribution" module (green/red tags).
- [ ] Integrate the "Activation patching" module (restoration bars).
- [ ] Integrate the "Weights" Bar chart (layer_0.mlp).

### 3.6 Tab: Evals
- [ ] Integrate the "Standardized benchmark" section (selection buttons + table).
- [ ] Integrate the "Output quality" section (text area + button + table).
- [ ] Integrate the "Export" section (Langfuse button + local path display).

## Phase 4: States, Interactions & Accessibility (a11y)
- [ ] Implement hover states on cards, buttons, and table rows.
- [ ] Implement focus states (blue border) for keyboard navigation.
- [ ] Create Skeleton screens (loading) for charts and tables.
- [ ] Create Empty state (welcome message if no dataset is loaded).
- [ ] Create Error state (banner if imported file is corrupted).
- [ ] Add tooltips on all charts (displaying exact values).
- [ ] Verify color contrasts (WCAG AA, 4.5:1 ratio for gray text).
- [ ] Add `aria-labels` on Heatmap cells (for screen readers and color-blind users).

## Phase 5: Responsive & QA
- [ ] **Tablet (768px - 1279px):** Make Sidebar collapsible, change KPI grids from 3 to 2 columns.
- [ ] **Mobile (<768px):** Transform Sidebar into a Drawer menu, stack charts vertically, make tables horizontally scrollable.
- [ ] Test file imports (CSV, JSON, JSONL).
- [ ] Test model switching via the global Dropdown.
- [ ] Perform final design review (Pixel Perfect) against mockups.
