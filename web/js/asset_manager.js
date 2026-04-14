/**
 * Copyright 2026 Google LLC
 * Built for AI Asset Tagging Extension
 */

import { app } from "../../../scripts/app.js";

function createAssetPanel() {
    // Container
    const panel = document.createElement("div");
    panel.id = "google-genai-asset-panel";
    panel.style.position = "fixed";
    panel.style.left = "0";
    panel.style.top = "50px";
    panel.style.bottom = "0";
    panel.style.width = "320px";
    panel.style.backgroundColor = "var(--comfy-menu-bg, #202020)";
    panel.style.color = "var(--fg-color, #ffffff)";
    panel.style.zIndex = "10000";
    panel.style.display = "flex";
    panel.style.flexDirection = "column";
    panel.style.fontFamily = "var(--font-family, sans-serif)";
    panel.style.borderRight = "1px solid var(--border-color, #404040)";
    panel.style.boxShadow = "4px 0 15px rgba(0, 0, 0, 0.5)";
    panel.style.transition = "transform 0.3s ease";
    panel.style.transform = "translateX(-100%)"; // Default hidden

    // Header
    const header = document.createElement("div");
    header.style.padding = "15px";
    header.style.borderBottom = "1px solid var(--border-color, #404040)";
    header.style.display = "flex";
    header.style.justifyContent = "space-between";
    header.style.alignItems = "center";

    const title = document.createElement("h3");
    title.innerText = "✨ GenAI Asset Explorer";
    title.style.margin = "0";
    title.style.fontSize = "1.1rem";
    title.style.fontWeight = "600";

    const closeBtn = document.createElement("button");
    closeBtn.innerText = "✕";
    closeBtn.style.background = "none";
    closeBtn.style.border = "none";
    closeBtn.style.color = "var(--input-text, #aaaaaa)";
    closeBtn.style.cursor = "pointer";
    closeBtn.style.fontSize = "1.2rem";
    closeBtn.onclick = () => {
        panel.style.transform = "translateX(-100%)";
    };

    header.appendChild(title);
    header.appendChild(closeBtn);
    panel.appendChild(header);

    // Controls
    const controls = document.createElement("div");
    controls.style.padding = "15px";
    controls.style.display = "flex";
    controls.style.flexDirection = "column";
    controls.style.gap = "10px";

    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.placeholder = "Search visually or by tags...";
    searchInput.style.padding = "10px";
    searchInput.style.borderRadius = "4px";
    searchInput.style.border = "1px solid var(--border-color, #404040)";
    searchInput.style.backgroundColor = "var(--comfy-input-bg, #1a1a1a)";
    searchInput.style.color = "var(--fg-color, #ffffff)";
    searchInput.style.outline = "none";

    const searchMode = document.createElement("select");
    searchMode.style.padding = "10px";
    searchMode.style.borderRadius = "4px";
    searchMode.style.border = "1px solid var(--border-color, #404040)";
    searchMode.style.backgroundColor = "var(--comfy-input-bg, #1a1a1a)";
    searchMode.style.color = "var(--fg-color, #ffffff)";
    const optSem = document.createElement("option");
    optSem.value = "semantic";
    optSem.innerText = "🔮 Semantic / Natural Language";
    const optTag = document.createElement("option");
    optTag.value = "tags";
    optTag.innerText = "🏷️ Strict Tags";
    searchMode.appendChild(optSem);
    searchMode.appendChild(optTag);

    const storageMode = document.createElement("select");
    storageMode.style.padding = "10px";
    storageMode.style.borderRadius = "4px";
    storageMode.style.border = "1px solid var(--border-color, #404040)";
    storageMode.style.backgroundColor = "var(--comfy-input-bg, #1a1a1a)";
    storageMode.style.color = "var(--fg-color, #ffffff)";
    const optLocal = document.createElement("option");
    optLocal.value = "local";
    optLocal.innerText = "📁 Storage: Local SQLite";
    const optCloud = document.createElement("option");
    optCloud.value = "cloud";
    optCloud.innerText = "☁️ Storage: GCP BigQuery / GCS";
    storageMode.appendChild(optLocal);
    storageMode.appendChild(optCloud);

    const searchBtn = document.createElement("button");
    searchBtn.innerText = "Explore Assets";
    searchBtn.style.padding = "10px";
    searchBtn.style.borderRadius = "4px";
    searchBtn.style.border = "1px solid var(--border-color, #404040)";
    searchBtn.style.backgroundColor = "var(--comfy-input-bg, #333)";
    searchBtn.style.color = "var(--fg-color, #ffffff)";
    searchBtn.style.fontWeight = "600";
    searchBtn.style.cursor = "pointer";
    searchBtn.style.transition = "background 0.2s";
    searchBtn.onmouseover = () => searchBtn.style.backgroundColor = "#444";
    searchBtn.onmouseout = () => searchBtn.style.backgroundColor = "var(--comfy-input-bg, #333)";

    controls.appendChild(searchInput);
    controls.appendChild(searchMode);
    controls.appendChild(storageMode);
    controls.appendChild(searchBtn);
    panel.appendChild(controls);

    // Gallery Container
    const gallery = document.createElement("div");
    gallery.style.flex = "1";
    gallery.style.overflowY = "auto";
    gallery.style.padding = "15px";
    gallery.style.display = "grid";
    gallery.style.gridTemplateColumns = "1fr";
    gallery.style.gap = "15px";
    panel.appendChild(gallery);

    // Pagination footer
    const footer = document.createElement("div");
    footer.style.padding = "10px 15px";
    footer.style.display = "flex";
    footer.style.justifyContent = "space-between";
    footer.style.alignItems = "center";
    footer.style.borderTop = "1px solid var(--border-color, #404040)";
    footer.style.backgroundColor = "var(--comfy-menu-bg, #202020)";

    const prevBtn = document.createElement("button");
    prevBtn.innerText = "⬅ Prev";
    prevBtn.style.background = "var(--comfy-input-bg, #1a1a1a)";
    prevBtn.style.border = "1px solid var(--border-color, #404040)";
    prevBtn.style.color = "var(--fg-color, #ffffff)";
    prevBtn.style.padding = "5px 10px";
    prevBtn.style.borderRadius = "4px";
    prevBtn.style.cursor = "pointer";

    const pageInfo = document.createElement("span");
    pageInfo.innerText = "Page 1";
    pageInfo.style.fontSize = "0.85rem";
    pageInfo.style.color = "var(--input-text, #aaaaaa)";

    const nextBtn = document.createElement("button");
    nextBtn.innerText = "Next ➡";
    nextBtn.style.background = "var(--comfy-input-bg, #1a1a1a)";
    nextBtn.style.border = "1px solid var(--border-color, #404040)";
    nextBtn.style.color = "var(--fg-color, #ffffff)";
    nextBtn.style.padding = "5px 10px";
    nextBtn.style.borderRadius = "4px";
    nextBtn.style.cursor = "pointer";

    footer.appendChild(prevBtn);
    footer.appendChild(pageInfo);
    footer.appendChild(nextBtn);
    panel.appendChild(footer);

    let currentPage = 1;
    const perPage = 3;

    prevBtn.onclick = () => {
        if (currentPage > 1) {
            currentPage--;
            loadAllAssets();
        }
    };

    nextBtn.onclick = () => {
        currentPage++;
        loadAllAssets();
    };

    storageMode.onchange = () => {
        currentPage = 1;
        loadAllAssets();
    };

    // Toggle Button (fixed on screen left)
    const toggleBtn = document.createElement("button");
    toggleBtn.innerText = "🔍 GenAI Assets";
    toggleBtn.style.position = "fixed";
    toggleBtn.style.left = "0";
    toggleBtn.style.top = "50%";
    toggleBtn.style.transform = "translateY(-50%)";
    toggleBtn.style.padding = "10px 5px";
    toggleBtn.style.writingMode = "vertical-rl";
    toggleBtn.style.textOrientation = "mixed";
    toggleBtn.style.backgroundColor = "var(--comfy-menu-bg, #202020)";
    toggleBtn.style.color = "var(--fg-color, #ffffff)";
    toggleBtn.style.border = "1px solid var(--border-color, #404040)";
    toggleBtn.style.borderLeft = "none";
    toggleBtn.style.borderTopRightRadius = "4px";
    toggleBtn.style.borderBottomRightRadius = "4px";
    toggleBtn.style.cursor = "pointer";
    toggleBtn.style.zIndex = "9999";
    toggleBtn.style.boxShadow = "2px 0 5px rgba(0,0,0,0.3)";

    toggleBtn.onclick = () => {
        if (panel.style.transform === "translateX(-100%)") {
            panel.style.transform = "translateX(0)";
            currentPage = 1;
            loadAllAssets();
        } else {
            panel.style.transform = "translateX(-100%)";
        }
    };
    document.body.appendChild(toggleBtn);
    document.body.appendChild(panel);

    async function loadAllAssets() {
        pageInfo.innerText = `Page ${currentPage}`;
        gallery.innerHTML = "<div style='color: var(--input-text, #aaaaaa); text-align: center;'>Loading assets...</div>";
        try {
            const res = await fetch(`/google_genmedia/asset_manager/list?page=${currentPage}&per_page=${perPage}&storage_mode=${storageMode.value}`);
            const data = await res.json();
            const items = data.assets || [];
            renderAssets(items);

            if (items.length < perPage) nextBtn.style.opacity = "0.5";
            else nextBtn.style.opacity = "1";

            if (currentPage === 1) prevBtn.style.opacity = "0.5";
            else prevBtn.style.opacity = "1";
        } catch (e) {
            gallery.innerHTML = `<div style='color: #ff4444;'>Error: ${e.message}</div>`;
        }
    }

    searchBtn.onclick = async () => {
        const query = searchInput.value.trim();
        if (!query) return loadAllAssets();

        gallery.innerHTML = "<div style='color: var(--input-text, #aaaaaa); text-align: center;'>Searching deep AI space...</div>";
        try {
            const res = await fetch("/google_genmedia/asset_manager/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, mode: searchMode.value, storage_mode: storageMode.value })
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            renderAssets(data.results || []);
        } catch (e) {
            gallery.innerHTML = `<div style='color: #ff4444;'>Error: ${e.message}</div>`;
        }
    };

    function renderAssets(items) {
        gallery.innerHTML = "";
        if (items.length === 0) {
            gallery.innerHTML = "<div style='color: var(--input-text, #aaaaaa); text-align: center;'>No relevant items found.</div>";
            return;
        }

        items.forEach(item => {
            const card = document.createElement("div");
            card.style.backgroundColor = "var(--comfy-input-bg, #1a1a1a)";
            card.style.borderRadius = "4px";
            card.style.overflow = "hidden";
            card.style.border = "1px solid var(--border-color, #404040)";
            card.style.display = "flex";
            card.style.flexDirection = "column";

            const imgContainer = document.createElement("div");
            imgContainer.style.height = "160px";
            imgContainer.style.overflow = "hidden";
            imgContainer.style.display = "flex";
            imgContainer.style.alignItems = "center";
            imgContainer.style.justifyContent = "center";
            imgContainer.style.backgroundColor = "#000000";

            let mediaElem;
            const isVideo = item.filepath && item.filepath.includes(".mp4");
            if (isVideo) {
                mediaElem = document.createElement("video");
                mediaElem.src = item.view_url || "";
                mediaElem.controls = true;
                mediaElem.autoplay = false;
                mediaElem.loop = true;
                mediaElem.playsInline = true;
                mediaElem.muted = true;
                mediaElem.style.width = "100%";
                mediaElem.style.height = "100%";
                mediaElem.style.objectFit = "cover";
            } else {
                mediaElem = document.createElement("img");
                mediaElem.src = item.view_url || "";
                mediaElem.style.width = "100%";
                mediaElem.style.height = "100%";
                mediaElem.style.objectFit = "cover";
            }
            imgContainer.appendChild(mediaElem);

            const meta = document.createElement("div");
            meta.style.padding = "10px";
            meta.style.display = "flex";
            meta.style.flexDirection = "column";
            meta.style.gap = "8px";

            const caption = document.createElement("div");
            caption.innerText = item.caption || "No caption available";
            caption.style.fontSize = "0.85rem";
            caption.style.color = "var(--input-text, #aaaaaa)";
            caption.style.maxHeight = "60px";
            caption.style.overflow = "hidden";
            caption.style.textOverflow = "ellipsis";

            const tags = document.createElement("div");
            tags.style.display = "flex";
            tags.style.flexWrap = "wrap";
            tags.style.gap = "4px";
            (item.tags || "").split(",").forEach(t => {
                const tStr = t.trim();
                if (!tStr) return;
                const badge = document.createElement("span");
                badge.innerText = tStr;
                badge.style.backgroundColor = "var(--comfy-menu-bg, #333)";
                badge.style.border = "1px solid var(--border-color, #555)";
                badge.style.color = "var(--fg-color, #ddd)";
                badge.style.padding = "2px 6px";
                badge.style.borderRadius = "4px";
                badge.style.fontSize = "0.7rem";
                tags.appendChild(badge);
            });

            const loadBtn = document.createElement("button");
            loadBtn.innerText = "📥 Load to Graph";
            loadBtn.style.padding = "6px";
            loadBtn.style.border = "1px solid var(--border-color, #555)";
            loadBtn.style.borderRadius = "4px";
            loadBtn.style.backgroundColor = "var(--comfy-menu-bg, #222)";
            loadBtn.style.color = "var(--fg-color, #fff)";
            loadBtn.style.fontWeight = "600";
            loadBtn.style.cursor = "pointer";
            loadBtn.style.fontSize = "0.8rem";
            loadBtn.onmouseover = () => loadBtn.style.backgroundColor = "#333";
            loadBtn.onmouseout = () => loadBtn.style.backgroundColor = "var(--comfy-menu-bg, #222)";
            loadBtn.onclick = async () => {
                try {
                    const res = await fetch("/google_genmedia/asset_manager/copy_to_input", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ filepath: item.filepath })
                    });
                    const data = await res.json();
                    if (data.error) throw new Error(data.error);

                    const isVid = item.filetype && (item.filetype.includes("video") || item.filetype.includes(".mp4"));
                    const nodeType = isVid ? "LoadVideo" : "LoadImage";

                    if (typeof LiteGraph !== "undefined" && app.graph) {
                        const node = LiteGraph.createNode(nodeType);
                        if (node) {
                            app.graph.add(node);

                            let px = 300;
                            let py = 300;
                            if (app.canvas && app.canvas.ds) {
                                px = -app.canvas.ds.offset[0] + 200;
                                py = -app.canvas.ds.offset[1] + 200;
                            }
                            node.pos = [px, py];

                            if (node.widgets) {
                                const widgetName = isVid ? "video" : "image";
                                const w = node.widgets.find(x => x.name === widgetName) || node.widgets[0];
                                if (w) w.value = data.filename;
                            }

                            if (app.canvas) app.canvas.setDirty(true, true);
                        }
                    }
                } catch (err) {
                    alert("Failed to load asset: " + err.message);
                }
            };

            meta.appendChild(caption);
            meta.appendChild(tags);
            meta.appendChild(loadBtn);

            card.appendChild(imgContainer);
            card.appendChild(meta);
            gallery.appendChild(card);
        });
    }
}

app.registerExtension({
    name: "google_genai.asset_manager",
    async setup() {
        createAssetPanel();
    }
});
