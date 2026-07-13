(function () {
  const chatWindow = document.getElementById('chat-window');
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const status = document.getElementById('chat-status');
  const chipsWrap = document.getElementById('suggested-chips');

  function addBubble(role, text) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble ' + role;

    const who = document.createElement('span');
    who.className = 'who';
    who.textContent = role === 'user' ? 'You' : 'AI Copilot';

    bubble.appendChild(who);
    bubble.appendChild(document.createTextNode(text));
    chatWindow.appendChild(bubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  async function sendMessage(overrideText) {
    const message = (overrideText !== undefined ? overrideText : input.value).trim();
    if (!message) return;

    addBubble('user', message);
    input.value = '';
    sendBtn.disabled = true;
    status.textContent = 'Retrieving relevant passages…';

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();

      if (!res.ok) {
        status.textContent = data.error || 'Something went wrong.';
      } else {
        addBubble('model', data.reply);
        status.textContent = '';
      }
    } catch (err) {
      status.textContent = 'Network error — please try again.';
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  sendBtn.addEventListener('click', () => sendMessage());
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendMessage();
    }
  });

  if (chipsWrap) {
    chipsWrap.addEventListener('click', (e) => {
      const chip = e.target.closest('.suggested-chip');
      if (!chip) return;
      sendMessage(chip.dataset.question);
    });
  }
})();
