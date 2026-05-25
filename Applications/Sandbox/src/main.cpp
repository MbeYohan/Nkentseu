// =============================================================================
// Sandbox/src/main.cpp  — Application de demonstration Nkentseu
// Pattern : Dispatcher evenementiel (push)
// =============================================================================

#include "NKWindow/Core/NkWindow.h"
#include "NKWindow/Core/NkSystem.h"
#include "NKWindow/Events/NkEventDispatcher.h"
#include "NKWindow/Events/NkEventSystem.h"
#include "NKWindow/Events/NkGamepadSystem.h"
#include "NKWindow/Core/NkMain.h"
#include "NKRenderer/NkRenderer.h"
#include "NKRenderer/NkRendererConfig.h"
#include "NKTime/NkChrono.h"

#include "NKLogger/NkLog.h"
#include "NKMemory/NkMemory.h"

#ifndef NK_SANDBOX_RENDERER_API
#define NK_SANDBOX_RENDERER_API nkentseu::NkRendererApi::NK_SOFTWARE
#endif

using namespace nkentseu;
using namespace nkentseu::math;

// =============================================================================
int nkmain(const nkentseu::NkEntryState& /*state*/)
{
    // -------------------------------------------------------------------------
    // 1. Initialisation du systeme
    // -------------------------------------------------------------------------
    if (!NkInitialise({ .appName = "AR From Scratch — Sandbox Demo" })) {
        logger.Error("[Sandbox] NkInitialise a echoue");
        return -1;
    }

    // -------------------------------------------------------------------------
    // 2. Creation de la fenetre
    // -------------------------------------------------------------------------
    NkWindowConfig wCfg;
    wCfg.title       = "AR From Scratch — Sandbox (Semaines 1-6)";
    wCfg.width       = 900;
    wCfg.height      = 600;
    wCfg.centered    = true;
    wCfg.resizable   = true;
    wCfg.dropEnabled = true;

    NkWindow window(wCfg);
    if (!window.IsOpen()) {
        logger.Error("[Sandbox] Echec creation fenetre");
        NkClose();
        return -2;
    }

    // -------------------------------------------------------------------------
    // 3. Renderer software
    // -------------------------------------------------------------------------
    NkRendererConfig rCfg;
    rCfg.api                   = NK_SANDBOX_RENDERER_API;
    rCfg.autoResizeFramebuffer = true;

    mem::NkUniquePtr<NkRenderer> renderer;
    if (rCfg.api != NkRendererApi::NK_NONE) {
        renderer = mem::NkMakeUnique<NkRenderer>();
        if (!renderer->Create(window, rCfg)) {
            logger.Error("[Sandbox] Echec creation renderer");
            NkClose();
            return -3;
        }
    }

    // -------------------------------------------------------------------------
    // 4. Boucle principale
    // -------------------------------------------------------------------------
    auto& evtSys = NkEvents();
    bool  alive  = true;
    NkChrono chrono;

    while (alive && window.IsOpen())
    {
        chrono.Reset();

        while (NkEvent* evt = evtSys.PollEvent())
            if (evt->Is<nkentseu::NkWindowCloseEvent>())
                alive = false;

        if (!alive || !window.IsOpen()) break;

        // Cap a 60 fps
        NkElapsedTime elapsed = chrono.Elapsed();
        if (elapsed.milliseconds < 16)
            NkChrono::Sleep(16 - elapsed.milliseconds);
        else
            NkChrono::YieldThread();
    }

    // -------------------------------------------------------------------------
    // 5. Nettoyage
    // -------------------------------------------------------------------------
    if (renderer) renderer->Shutdown();
    window.Close();
    NkClose();
    return 0;
}
