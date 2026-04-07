/* OpenCrust docs — branding & polish */
(function () {
  "use strict";

  /* ── Logo in sidebar ── */
  function injectLogo() {
    var scrollbox = document.querySelector(".sidebar .sidebar-scrollbox");
    if (!scrollbox) return;
    if (scrollbox.querySelector(".oc-logo")) return;

    var wrap = document.createElement("a");
    wrap.href = "index.html";
    wrap.className = "oc-logo";
    wrap.style.cssText = [
      "display:flex",
      "align-items:center",
      "gap:10px",
      "padding:18px 16px 14px",
      "border-bottom:1px solid rgba(232,101,26,0.2)",
      "margin-bottom:8px",
      "text-decoration:none",
    ].join(";");

    var img = document.createElement("img");
    img.src =
      "https://github.com/opencrust-org/opencrust/raw/main/assets/logo.png";
    img.alt = "OpenCrust";
    img.style.cssText =
      "height:32px;width:auto;filter:drop-shadow(0 0 6px rgba(232,101,26,0.4))";
    img.onerror = function () {
      this.style.display = "none";
      label.textContent = "🦀 OpenCrust";
    };

    var label = document.createElement("span");
    label.textContent = "OpenCrust";
    label.style.cssText = [
      "font-family:'Clash Display',sans-serif",
      "font-weight:700",
      "font-size:1.1rem",
      "color:#e8651a",
      "letter-spacing:0.04em",
    ].join(";");

    wrap.appendChild(img);
    wrap.appendChild(label);
    scrollbox.insertBefore(wrap, scrollbox.firstChild);
  }

  /* ── Traffic-light dots on code blocks ── */
  function styleCodeBlocks() {
    document.querySelectorAll("pre").forEach(function (pre) {
      if (pre.querySelector(".oc-dots")) return;

      var dots = document.createElement("span");
      dots.className = "oc-dots";
      dots.setAttribute("aria-hidden", "true");
      dots.style.cssText = [
        "position:absolute",
        "top:10px",
        "left:14px",
        "display:flex",
        "gap:5px",
        "z-index:5",
      ].join(";");

      [
        ["#ff5f57", "#c0392b"],
        ["#febc2e", "#b7870d"],
        ["#28c840", "#1a7a28"],
      ].forEach(function (c) {
        var dot = document.createElement("span");
        dot.style.cssText = [
          "width:11px",
          "height:11px",
          "border-radius:50%",
          "background:" + c[0],
          "box-shadow:0 0 4px " + c[1],
          "display:inline-block",
        ].join(";");
        dots.appendChild(dot);
      });

      pre.style.position = "relative";
      pre.insertBefore(dots, pre.firstChild);

      /* remove CSS fallback text "● ● ●" once real dots are added */
      pre.style.setProperty("--oc-dots-injected", "1");
    });
  }

  /* ── Fade-in content on load ── */
  function animateContent() {
    var main = document.querySelector(".content main");
    if (!main) return;
    main.style.opacity = "0";
    main.style.transform = "translateY(12px)";
    requestAnimationFrame(function () {
      main.style.transition = "opacity 0.5s ease, transform 0.5s ease";
      main.style.opacity = "1";
      main.style.transform = "translateY(0)";
    });
  }

  /* ── Init ── */
  function init() {
    injectLogo();
    styleCodeBlocks();
    animateContent();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  /* re-run on mdBook page navigation (SPA-style) */
  var observer = new MutationObserver(function (mutations) {
    mutations.forEach(function () {
      styleCodeBlocks();
    });
  });
  var content = document.getElementById("content");
  if (content) {
    observer.observe(content, { childList: true, subtree: true });
  }
})();
