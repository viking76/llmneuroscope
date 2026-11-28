# Tasks Backlog — LLM Neuroscope (MVP)

**Project:** LLM Neuroscope  
**Sprint:** MVP Development  
**Status:** To Do

---

## Epic 1: Design System & UI Foundation
- [ ] **TASK-001: Setup Design Tokens (Colors)**
  - Configure CSS variables/Tailwind for dark theme (`bg-global`, `bg-surface`, semantic colors, heatmap gradients).
- [ ] **TASK-002: Setup Typography**
  - Configure Inter/Roboto fonts, weights (Regular, Semi-bold, Bold), and sizes (H1, H2, Body, KPI).
- [ ] **TASK-003: Build Core UI Components**
  - Create reusable components for Buttons (Primary, Secondary, Danger), Badges (Success, Warning, Danger), Inputs, and Cards.

## Epic 2: Global Layout & Navigation
- [ ] **TASK-004: Build Sidebar Component**
  - Implement file import zone (Drag & Drop + Button) and dataset list display.
- [ ] **TASK-005: Build Global Header Component**
  - Implement App Title, Dataset counters, Tab navigation, Model selector dropdown, and Export buttons (CSV, PDF).
- [ ] **TASK-006: Implement App Routing**
  - Set up routing state to toggle between the 6 main tabs without page reloads.

## Epic 3: Screen Development (Views)
### Overview & Activity
- [ ] **TASK-007: Develop Overview KPI Cards**
  - Build the 6 KPI cards (Perplexity, Token entropy, Degeneracy rate, Max drift, Avg length, Avg latency).
- [ ] **TASK-008: Develop Overview Charts**
  - Implement Bar chart (Entropy per run) and Line chart (Log-prob & Entropy per position) with toggleable legends.
- [ ] **TASK-009: Develop Activity Heatmap**
  - Implement the Activity Heatmap (Entropy per position x run) with the blue-to-red gradient and horizontal legend.
- [ ] **TASK-010: Develop Activity Histogram**
  - Implement the Bar chart for the Entropy profile per position.

### Anomalies & Comparison
- [ ] **TASK-011: Develop Anomalies View**
  - Build the 4 KPI cards, Table 1 (Anomaly log with severity badges), and Table 2 (Blocking points).
- [ ] **TASK-012: Add "View Prompt" Action to Anomalies**
  - Add an interaction icon/button on anomaly rows to view the exact prompt text (UX improvement).
- [ ] **TASK-013: Develop Comparison Header**
  - Build model selectors (A/B), global status badge ("IMPROVEMENT"), and divergence metric display.
- [ ] **TASK-014: Develop Comparison Data Table**
  - Build the metrics table (Metric, Model A, Model B, Delta, Delta %, Status) and implement column sorting.

### Neuron Probe & Evals
- [ ] **TASK-015: Setup Neuron Probe Layout**
  - Implement accordion or sub-tabs to prevent cognitive overload for the dense visualization page.
- [ ] **TASK-016: Develop Neuron Probe - Generation & Activation**
  - Build Instrumented Generation (Input, Output, Perplexity) and Layer Activation analysis (Progress bars, badges).
- [ ] **TASK-017: Develop Neuron Probe - Attention & Tokens**
  - Build Attention Entropy heatmap (Layer x Head) and Token Attribution module (green/red tags).
- [ ] **TASK-018: Develop Neuron Probe - Patching & Weights**
  - Build Activation Patching progress bars and the Weights Bar chart (layer_0.mlp).
- [ ] **TASK-019: Develop Evals View**
  - Build Standardized Benchmark section (selector buttons, table) and Output Quality section (text area, evaluation table).
- [ ] **TASK-020: Develop Evals Export Section**
  - Add "Push to Langfuse" button and local path display.

## Epic 4: System States, Interactions & Accessibility (a11y)
- [ ] **TASK-021: Implement Interactive States**
  - Add Hover states (cards, rows) and Focus states (blue borders for keyboard navigation).
- [ ] **TASK-022: Implement System States**
  - Design and code Loading (Skeleton screens), Empty states (no dataset uploaded), and Error states (corrupted file upload).
- [ ] **TASK-023: Implement Chart Tooltips**
  - Ensure all charts (Bar, Line, Heatmap cells) display exact data values on hover.
- [ ] **TASK-024: Accessibility Compliance**
  - Verify color contrasts (WCAG AA, 4.5:1 ratio for gray text) and add `aria-labels` to Heatmap cells for color-blind users.

## Epic 5: Responsive Design & QA
- [ ] **TASK-025: Implement Tablet Layout**
  - Collapse Sidebar into a toggleable menu, adjust KPI grids from 3 columns to 2.
- [ ] **TASK-026: Implement Mobile Layout**
  - Transform Sidebar into a Drawer menu, stack all charts vertically, and make data tables horizontally scrollable.
- [ ] **TASK-027: QA - File Upload & Parsing**
  - Test the import functionality with valid and invalid CSV, JSON, and JSONL files.
- [ ] **TASK-028: QA - Final Design Review**
  - Conduct a pixel-perfect review against Figma mockups and ensure the model selector updates data across all tabs correctly.
