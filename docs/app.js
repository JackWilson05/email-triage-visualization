/* global d3 */
(async function () {
  // Blue and red always encode the email's actual class. The dashboard itself uses slate neutrals.
  const colors = { ham: "#3d6f9e", spam: "#c65a52", amber: "#b8892d", ink: "#243447", slate: "#526779", muted: "#5e6d7f", line: "#d8e0e8" };
  const tooltip = d3.select("#tooltip");
  const formatPct = d3.format(".1%");
  const formatNum = d3.format(",");

  function tip(html, event) {
    tooltip.html(html).classed("visible", true)
      .style("left", `${Math.min(window.innerWidth - 285, event.clientX + 14)}px`)
      .style("top", `${Math.max(12, event.clientY - 16)}px`);
  }
  // D3 v5 passes the datum first and stores the pointer event on d3.event;
  // D3 v6+ passes (event, datum). Supporting both keeps the offline copy safe.
  function eventAndDatum(first, second) {
    return first && Number.isFinite(first.clientX) ? [first, second] : [d3.event, first];
  }
  function hideTip() { tooltip.classed("visible", false); }
  function chartFrame(selector, margins = { top: 20, right: 24, bottom: 48, left: 58 }) {
    const node = document.querySelector(selector);
    const width = Math.max(300, node.clientWidth);
    const height = Math.max(330, node.clientHeight || 360);
    const svg = d3.select(node).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", height);
    return { svg, width, height, margins, innerWidth: width - margins.left - margins.right, innerHeight: height - margins.top - margins.bottom };
  }
  function key(value) { return String(value).replaceAll(" ", "_").replaceAll("|", "").toLowerCase(); }
  function curveStyle(name) {
    const featureSet = name.includes("combined") ? "combined" : name.includes("context") ? "context" : "text";
    return {
      color: name.includes("naive_bayes") ? colors.ink : colors.slate,
      dash: featureSet === "combined" ? "2 3" : featureSet === "context" ? "7 4" : null,
    };
  }

  let data = window.WEBSITE_DATA;
  try {
    if (!data) data = await d3.json("data/website_data.json");
  } catch (error) {
    document.body.insertAdjacentHTML("afterbegin", "<p style='padding:1rem;background:#fff0ed;color:#8c342e'>The site data file is missing. Run the modeling script, then make sure both website_data.json and website_data.js are inside docs/data/.</p>");
    throw error;
  }
  const { summary, metrics, points, distributions, coverage, pr_curves: curves } = data;
  const warningTarget = summary.warning_precision_target ?? 0.75;
  const blockTarget = summary.block_precision_target ?? 0.95;
  const selectedModel = metrics.find(d => d.model === summary.best_model);

  d3.select("#summary-grid").selectAll("div").data([
    [formatNum(summary.joined_records), "verified joined emails"],
    [formatPct(summary.spam_rate), "spam prevalence"],
    ["6", "shared-input model variants"],
    [`${formatPct(selectedModel.block_precision_test)} / ${formatPct(blockTarget)}`, "observed / target block precision"],
  ]).join("div").attr("class", "summary-card").html(d => `<strong>${d[0]}</strong><span>${d[1]}</span>`);

  // 01: field coverage
  {
    const c = chartFrame("#coverage-chart", { top: 18, right: 70, bottom: 38, left: 130 });
    const fields = coverage.map(d => ({ ...d, label: d.field.replaceAll("_", " ") }));
    const x = d3.scaleLinear().domain([0, summary.joined_records]).range([0, c.innerWidth]);
    const y = d3.scaleBand().domain(fields.map(d => d.label)).range([0, c.innerHeight]).padding(.28);
    const g = c.svg.append("g").attr("transform", `translate(${c.margins.left},${c.margins.top})`);
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickSize(0));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${c.innerHeight})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format("~s")));
    g.selectAll("rect").data(fields).join("rect").attr("y", d => y(d.label)).attr("height", y.bandwidth()).attr("width", d => x(d.available)).attr("fill", colors.slate).attr("rx", 3)
      .on("mousemove", (event, d) => { [event, d] = eventAndDatum(event, d); tip(`<b>${d.label}</b><br>${formatNum(d.available)} of ${formatNum(summary.joined_records)} emails had a parsed value.`, event); }).on("mouseleave", hideTip);
    g.selectAll("text.count").data(fields).join("text").attr("class", "legend").attr("x", d => x(d.available) + 7).attr("y", d => y(d.label) + y.bandwidth() / 2 + 4).text(d => formatNum(d.available));
  }

  // 02: held-out distributions as small-multiple boxplots
  {
    const c = chartFrame("#distribution-chart", { top: 34, right: 26, bottom: 28, left: 24 });
    const features = ["url_count", "text_word_count", "received_hops", "attachment_count"];
    const featureLabels = {
      url_count: "URL count", text_word_count: "Text word count",
      received_hops: "Received hops", attachment_count: "Attachment count",
    };
    const panelColumns = c.width < 620 ? 1 : 2;
    const panelWidth = c.innerWidth / panelColumns;
    const panelHeight = c.innerHeight / Math.ceil(features.length / panelColumns);

    function boxStats(values) {
      const sorted = values.filter(Number.isFinite).sort(d3.ascending);
      const q1 = d3.quantile(sorted, .25);
      const median = d3.quantile(sorted, .5);
      const q3 = d3.quantile(sorted, .75);
      const iqr = q3 - q1;
      const lower = d3.min(sorted.filter(value => value >= q1 - 1.5 * iqr));
      const upper = d3.max(sorted.filter(value => value <= q3 + 1.5 * iqr));
      return { q1, median, q3, lower, upper, count: sorted.length };
    }

    features.forEach((feature, index) => {
      const column = index % panelColumns;
      const row = Math.floor(index / panelColumns);
      const panel = c.svg.append("g").attr("transform", `translate(${c.margins.left + column * panelWidth},${c.margins.top + row * panelHeight})`);
      const values = points.map(point => +point[feature]).filter(Number.isFinite).sort(d3.ascending);
      const domain = [d3.quantile(values, .02), d3.quantile(values, .98)];
      const x = d3.scaleLinear().domain(domain[0] === domain[1] ? [0, domain[1] + 1] : domain).nice().range([46, panelWidth - 20]);
      const y = d3.scalePoint().domain(["Legitimate", "Spam"]).range([52, panelHeight - 45]);
      const groups = [
        { label: "Legitimate", labelValue: 0, color: colors.ham },
        { label: "Spam", labelValue: 1, color: colors.spam },
      ].map(group => ({ ...group, stats: boxStats(points.filter(point => point.label === group.labelValue).map(point => +point[feature])) }));

      panel.append("text").attr("x", 0).attr("y", 14).attr("fill", colors.ink).attr("font-weight", 700).attr("font-size", ".94rem").text(featureLabels[feature]);
      panel.append("g").attr("class", "axis").attr("transform", `translate(0,${panelHeight - 22})`).call(d3.axisBottom(x).ticks(4));
      panel.append("g").attr("class", "axis").attr("transform", "translate(42,0)").call(d3.axisLeft(y).tickSize(0));

      groups.forEach(group => {
        const yMid = y(group.label);
        const boxHeight = 34;
        const g = panel.append("g");
        g.append("line").attr("x1", x(group.stats.lower)).attr("x2", x(group.stats.upper)).attr("y1", yMid).attr("y2", yMid).attr("stroke", group.color).attr("stroke-width", 2);
        g.append("line").attr("x1", x(group.stats.lower)).attr("x2", x(group.stats.lower)).attr("y1", yMid - 9).attr("y2", yMid + 9).attr("stroke", group.color).attr("stroke-width", 2);
        g.append("line").attr("x1", x(group.stats.upper)).attr("x2", x(group.stats.upper)).attr("y1", yMid - 9).attr("y2", yMid + 9).attr("stroke", group.color).attr("stroke-width", 2);
        g.append("rect").attr("x", x(group.stats.q1)).attr("y", yMid - boxHeight / 2).attr("width", Math.max(2, x(group.stats.q3) - x(group.stats.q1))).attr("height", boxHeight).attr("fill", group.color).attr("fill-opacity", .42).attr("stroke", group.color).attr("stroke-width", 2)
          .on("mousemove", (event, d) => { [event] = eventAndDatum(event, d); tip(`<b>${featureLabels[feature]} · ${group.label}</b><br>Median: ${d3.format(".3~g")(group.stats.median)}<br>Middle 50%: ${d3.format(".3~g")(group.stats.q1)}–${d3.format(".3~g")(group.stats.q3)}<br>Whiskers: ${d3.format(".3~g")(group.stats.lower)}–${d3.format(".3~g")(group.stats.upper)}<br>n = ${formatNum(group.stats.count)}`, event); }).on("mouseleave", hideTip);
        g.append("line").attr("x1", x(group.stats.median)).attr("x2", x(group.stats.median)).attr("y1", yMid - boxHeight / 2).attr("y2", yMid + boxHeight / 2).attr("stroke", colors.ink).attr("stroke-width", 2.5);
      });
    });
    c.svg.append("text").attr("class", "legend").attr("x", c.margins.left).attr("y", c.height - 4).text("Box = middle 50%; center line = median; whiskers extend to the last value within 1.5 × IQR. Each panel has its own scale.");
  }

  // 03: score agreement scatter
  {
    const c = chartFrame("#risk-chart", { top: 18, right: 28, bottom: 52, left: 60 });
    const x = d3.scaleLinear().domain([0, 1]).range([0, c.innerWidth]);
    const y = d3.scaleLinear().domain([0, 1]).range([c.innerHeight, 0]);
    const g = c.svg.append("g").attr("transform", `translate(${c.margins.left},${c.margins.top})`);
    g.append("g").attr("class", "grid").call(d3.axisLeft(y).tickSize(-c.innerWidth).tickFormat(""));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${c.innerHeight})`).call(d3.axisBottom(x).ticks(5));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    g.append("line").attr("x1", x(.5)).attr("x2", x(.5)).attr("y2", c.innerHeight).attr("stroke", colors.line).attr("stroke-dasharray", "4 4");
    g.append("line").attr("x2", c.innerWidth).attr("y1", y(.5)).attr("y2", y(.5)).attr("stroke", colors.line).attr("stroke-dasharray", "4 4");
    g.selectAll("circle.point").data(points).join("circle").attr("class", "point").attr("cx", d => x(d.score_xgboost_text)).attr("cy", d => y(d.score_xgboost_context)).attr("r", 3.2).attr("fill", d => d.label ? colors.spam : colors.ham).attr("fill-opacity", .52)
      .on("mousemove", (event, d) => { [event, d] = eventAndDatum(event, d); tip(`<b>${d.label ? "Spam" : "Legitimate"} · ${d.error_type}</b><br>Text score: ${formatPct(d.score_xgboost_text)}<br>Context score: ${formatPct(d.score_xgboost_context)}<br>URLs: ${d.url_count}; Received hops: ${d.received_hops}<br>HTML: ${d.is_html ? "yes" : "no"}; attachment: ${d.has_attachment ? "yes" : "no"}`, event); }).on("mouseleave", hideTip);
    c.svg.append("text").attr("class", "legend").attr("x", c.margins.left + c.innerWidth / 2).attr("y", c.height - 10).attr("text-anchor", "middle").text("Text-only XGBoost score");
    c.svg.append("text").attr("class", "legend").attr("transform", `translate(15,${c.margins.top + c.innerHeight / 2}) rotate(-90)`).attr("text-anchor", "middle").text("Context-only XGBoost score");
  }

  // 04a: six-model bar chart
  {
    const c = chartFrame("#model-chart", { top: 18, right: 26, bottom: 84, left: 48 });
    const ordered = [...metrics].sort((a, b) => b.test_pr_auc - a.test_pr_auc);
    const x = d3.scaleBand().domain(ordered.map(d => d.model)).range([0, c.innerWidth]).padding(.22);
    const y = d3.scaleLinear().domain([0, 1]).nice().range([c.innerHeight, 0]);
    const g = c.svg.append("g").attr("transform", `translate(${c.margins.left},${c.margins.top})`);
    g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(5).tickSize(-c.innerWidth).tickFormat(""));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat(formatPct));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${c.innerHeight})`).call(d3.axisBottom(x).tickFormat(d => d.replace("Naive Bayes | ", "NB · ").replace("XGBoost | ", "XGB · "))).selectAll("text").attr("transform", "rotate(-32)").style("text-anchor", "end");
    g.selectAll("rect").data(ordered).join("rect").attr("x", d => x(d.model)).attr("y", d => y(d.test_pr_auc)).attr("height", d => c.innerHeight - y(d.test_pr_auc)).attr("width", x.bandwidth()).attr("fill", colors.slate).attr("rx", 3)
      .on("mousemove", (event, d) => { [event, d] = eventAndDatum(event, d); tip(`<b>${d.model}</b><br>Test average precision: ${formatPct(d.test_pr_auc)}<br>Test ROC AUC: ${formatPct(d.test_roc_auc)}<br>${formatPct(blockTarget)} band: ${formatPct(d.block_precision_test)} precision, ${formatPct(d.block_recall_test)} recall`, event); }).on("mouseleave", hideTip);
  }

  // 04b: PR curves
  {
    const c = chartFrame("#pr-chart", { top: 18, right: 22, bottom: 72, left: 52 });
    const g = c.svg.append("g").attr("transform", `translate(${c.margins.left},${c.margins.top})`);
    const x = d3.scaleLinear().domain([0, 1]).range([0, c.innerWidth]);
    const y = d3.scaleLinear().domain([0, 1]).range([c.innerHeight, 0]);
    g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(5).tickSize(-c.innerWidth).tickFormat(""));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat(formatPct));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${c.innerHeight})`).call(d3.axisBottom(x).ticks(5).tickFormat(formatPct));
    const line = d3.line().x(d => x(d.recall)).y(d => y(d.precision));
    Object.entries(curves).forEach(([name, curve]) => {
      const display = name.replaceAll("_", " ");
      const style = curveStyle(name);
      g.append("path").datum(curve).attr("fill", "none").attr("stroke", style.color).attr("stroke-width", 2).attr("stroke-dasharray", style.dash).attr("d", line)
        .on("mousemove", event => { [event] = eventAndDatum(event); tip(`<b>${display}</b><br>Precision-recall curve. Color indicates algorithm; solid, dashed, and dotted lines indicate text, context, and combined inputs.`, event); }).on("mouseleave", hideTip);
    });
    g.append("text").attr("class", "legend").attr("x", 0).attr("y", c.innerHeight + 48).text("Dark slate: Naive Bayes   ·   Medium slate: XGBoost");
    g.append("text").attr("class", "legend").attr("x", 0).attr("y", c.innerHeight + 66).text("Solid: text   ·   Dashed: context   ·   Dotted: combined");
  }

  // 05: threshold policy check on held-out data
  {
    const policyGroups = [
      { label: "Deliver" },
      { label: "Warn or review" },
      { label: "Quarantine" },
    ].map(group => {
      const rows = points.filter(point => point.triage_band === group.label);
      const spam = rows.filter(point => point.label === 1).length;
      let cursor = 0;
      const segments = [0, 1].map(label => {
        const values = rows.filter(point => point.label === label);
        const segment = { label, count: values.length, start: cursor };
        cursor += values.length;
        return segment;
      }).filter(segment => segment.count);
      return { ...group, count: rows.length, spam, legitimate: rows.length - spam, precision: rows.length ? spam / rows.length : 0, segments };
    });
    const deliver = policyGroups[0];
    const review = policyGroups[1];
    const quarantine = policyGroups[2];
    d3.select("#policy-summary").attr("class", "policy-summary").html(`
      <div><strong>${formatNum(deliver.count)} delivered</strong><span>${formatNum(deliver.legitimate)} legitimate; ${formatNum(deliver.spam)} spam.</span></div>
      <div><strong>${formatNum(review.count)} reviewed</strong><span>${formatNum(review.legitimate)} legitimate; ${formatNum(review.spam)} spam.</span></div>
      <div><strong>${formatNum(quarantine.count)} quarantined</strong><span>${formatNum(quarantine.legitimate)} legitimate; ${formatNum(quarantine.spam)} spam.</span></div>`);
    const compactPolicy = document.querySelector("#policy-chart").clientWidth < 600;
    const c = chartFrame("#policy-chart", { top: 64, right: compactPolicy ? 16 : 28, bottom: 32, left: compactPolicy ? 108 : 130 });
    const x = d3.scaleLinear().domain([0, d3.max(policyGroups, d => d.count)]).nice().range([0, c.innerWidth]);
    const y = d3.scaleBand().domain(policyGroups.map(d => d.label)).range([0, c.innerHeight - 48]).padding(.33);
    const g = c.svg.append("g").attr("transform", `translate(${c.margins.left},${c.margins.top})`);
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickSize(0));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${c.innerHeight - 48})`).call(d3.axisBottom(x).ticks(5).tickFormat(formatNum));
    const barGroups = g.selectAll("g.policy-group").data(policyGroups).join("g").attr("class", "policy-group").attr("transform", d => `translate(0,${y(d.label)})`);
    barGroups.each(function (group) {
      d3.select(this).selectAll("rect.policy").data(group.segments).join("rect")
        .attr("class", "policy").attr("x", segment => x(segment.start))
        .attr("y", 0).attr("width", segment => x(segment.count)).attr("height", y.bandwidth()).attr("stroke", "white").attr("stroke-width", 1)
        .attr("fill", segment => segment.label === 1 ? colors.spam : colors.ham)
        .on("mousemove", (event, segment) => { [event, segment] = eventAndDatum(event, segment); const truth = segment.label ? "spam" : "legitimate"; tip(`<b>${group.label}</b><br>${formatNum(segment.count)} ${truth} emails<br>Color shows the true class, not whether the action was correct.`, event); }).on("mouseleave", hideTip);
    });
    barGroups.append("text").attr("class", "bar-label").attr("x", 10).attr("y", y.bandwidth() / 2 + 4).attr("fill", "#ffffff").attr("font-size", ".76rem").attr("font-weight", 700).text(d => compactPolicy ? `${formatNum(d.count)} emails` : `${formatNum(d.count)} emails · ${formatPct(d.precision)} spam`);
    g.append("rect").attr("x", 0).attr("y", -48).attr("width", 18).attr("height", 12).attr("fill", colors.ham);
    g.append("text").attr("class", "legend").attr("x", 24).attr("y", -38).text("Legitimate / ham");
    g.append("rect").attr("x", 148).attr("y", -48).attr("width", 18).attr("height", 12).attr("fill", colors.spam);
    g.append("text").attr("class", "legend").attr("x", 172).attr("y", -38).text("Spam");
    g.append("text").attr("class", "legend").attr("x", 0).attr("y", -17).text("Ideal: blue only in Deliver; red only in Quarantine; no review band.");
  }

  d3.select("#python-figures").selectAll("a").data([
    ["01_six_model_average_precision.png", "Six-model comparison"],
    ["02_selected_signal_distributions.png", "Signal distributions"],
    ["03_text_context_risk_map.png", "Text/context risk map"],
    ["04_precision_recall_curves.png", "Precision-recall curves"],
    ["05_precision_based_triage_thresholds.png", "Triage thresholds"],
  ]).join("a").attr("href", d => `assets/${d[0]}`).attr("target", "_blank").attr("rel", "noreferrer").text(d => d[1]);
})();
