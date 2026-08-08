(function () {
  const root = document.documentElement;
  const stored = localStorage.getItem("spas-theme");
  if (stored) root.setAttribute("data-theme", stored);

  document.addEventListener("DOMContentLoaded", function () {
    const toggle = document.getElementById("themeToggle");
    if (toggle) {
      toggle.addEventListener("click", function () {
        const current = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", current);
        localStorage.setItem("spas-theme", current);
      });
    }

    const burger = document.getElementById("sidebarToggle");
    const sidebar = document.getElementById("sidebar");
    const backdrop = document.getElementById("sidebarBackdrop");

    function openSidebar() {
      sidebar.classList.add("open");
      if (backdrop) backdrop.classList.add("open");
    }
    function closeSidebar() {
      sidebar.classList.remove("open");
      if (backdrop) backdrop.classList.remove("open");
    }

    if (burger && sidebar) {
      burger.addEventListener("click", function () {
        sidebar.classList.contains("open") ? closeSidebar() : openSidebar();
      });
    }
    if (backdrop) {
      backdrop.addEventListener("click", closeSidebar);
    }
    if (sidebar) {
      // Tapping a nav link on mobile should close the drawer instead of leaving it open
      // over the newly-loaded page.
      sidebar.querySelectorAll("a.nav-link").forEach(function (link) {
        link.addEventListener("click", closeSidebar);
      });
    }

    localizeTimes();
  });

  // Every timestamp in this app is stored and rendered server-side in UTC
  // (data-utc="...Z"). The server doesn't know each visitor's real timezone,
  // so we convert to the browser's actual local time here, client-side --
  // this is correct regardless of where the person viewing the page is,
  // without needing a stored per-user timezone setting.
  function localizeTimes() {
    document.querySelectorAll(".local-time[data-utc]").forEach(function (el) {
      const iso = el.getAttribute("data-utc");
      if (!iso) return;
      const date = new Date(iso);
      if (isNaN(date.getTime())) return;

      const fmt = el.getAttribute("data-fmt") || "datetime";
      let text;
      if (fmt === "date") {
        text = date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
      } else if (fmt === "datetime-sec") {
        text = date.toLocaleString(undefined, {
          year: "numeric", month: "short", day: "numeric",
          hour: "2-digit", minute: "2-digit", second: "2-digit",
        });
      } else {
        text = date.toLocaleString(undefined, {
          year: "numeric", month: "short", day: "numeric",
          hour: "2-digit", minute: "2-digit",
        });
      }
      el.textContent = text;
    });
  }
})();
