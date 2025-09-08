package tui

import (
	"fmt"
	"net/url"
	"strings"
	"time"

	"two-factor-auth/account"
	"two-factor-auth/crypto"
	"two-factor-auth/storage"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/pquerna/otp/totp"
)

var (
	docStyle = lipgloss.NewStyle().Margin(1, 2)

	titleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("62")).
			Bold(true).
			Margin(0, 0, 1, 0)

	listContainerStyle = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("244")).
				Padding(1, 2)

	headerStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("244")).
			Bold(true).
			Padding(0, 0, 1, 0)

	issuerStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("242")).Width(20)
	accountNameStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Width(26)
	codeStyle        = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Bold(true).Width(10)

	countdownBaseStyle     = lipgloss.NewStyle().Width(10)
	countdownNormalStyle   = countdownBaseStyle.Copy().Foreground(lipgloss.Color("70"))
	countdownWarningStyle  = countdownBaseStyle.Copy().Foreground(lipgloss.Color("214"))
	countdownCriticalStyle = countdownBaseStyle.Copy().Foreground(lipgloss.Color("196")).Bold(true)

	helpStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("241")).Margin(1, 0)

	errorStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("197")).Bold(true)
	inputPromptStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("81")).Bold(true)
)

const totpPeriod = 30

type AppState int

const (
	stateViewingCodes AppState = iota
	stateAddingAccount
)

type model struct {
	state         AppState
	accounts      []account.Account
	codes         map[string]string
	remaining     int
	textInput     textinput.Model
	err           error
	encryptionKey []byte
	storagePath   string
}

type tickMsg time.Time
type codesGeneratedMsg struct{ codes map[string]string }
type errorMsg struct{ err error }
type accountAddedMsg struct{ newAccount account.Account }

func Start(key []byte, accs []account.Account, path string) error {
	ti := textinput.New()
	ti.Placeholder = "otpauth://..."
	ti.Focus()
	ti.CharLimit = 256
	ti.Width = 60

	initialState := stateViewingCodes
	if len(accs) == 0 {
		initialState = stateAddingAccount
	}

	m := model{
		state:         initialState,
		accounts:      accs,
		codes:         make(map[string]string),
		remaining:     totpPeriod,
		textInput:     ti,
		encryptionKey: key,
		storagePath:   path,
	}

	p := tea.NewProgram(m, tea.WithAltScreen())
	_, err := p.Run()
	return err
}

func (m model) Init() tea.Cmd {
	return tea.Batch(m.generateCodes(), doTick())
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch m.state {
		case stateViewingCodes:
			switch msg.String() {
			case "q", "ctrl+c":
				return m, tea.Quit
			case "a":
				m.state = stateAddingAccount
				m.textInput.Reset()
				return m, textinput.Blink
			}
		case stateAddingAccount:
			switch msg.String() {
			case "ctrl+c", "esc":
				m.state = stateViewingCodes
				m.err = nil
				return m, nil
			case "enter":
				cmds = append(cmds, m.addNewAccountCmd(m.textInput.Value()))
			}
		}
	case tickMsg:
		m.remaining = totpPeriod - (time.Now().Second() % totpPeriod)
		if m.remaining == totpPeriod {
			cmds = append(cmds, m.generateCodes())
		}
		cmds = append(cmds, doTick())

	case codesGeneratedMsg:
		m.codes = msg.codes
	case errorMsg:
		m.err = msg.err
	case accountAddedMsg:
		m.accounts = append(m.accounts, msg.newAccount)
		m.state = stateViewingCodes
		m.err = nil
		cmds = append(cmds, m.generateCodes())
	}

	if m.state == stateAddingAccount {
		m.textInput, cmd = m.textInput.Update(msg)
		cmds = append(cmds, cmd)
	}

	return m, tea.Batch(cmds...)
}

func (m model) View() string {
	var finalView string

	switch m.state {
	case stateViewingCodes:
		var listBuilder strings.Builder

		header := lipgloss.JoinHorizontal(lipgloss.Left,
			issuerStyle.Render("ISSUER"),
			accountNameStyle.Render("ACCOUNT"),
			codeStyle.Render("CODE"),
			countdownBaseStyle.Render("EXPIRES"),
		)
		listBuilder.WriteString(headerStyle.Render(header))
		listBuilder.WriteString("\n")

		for i, acc := range m.accounts {
			issuer := truncateString(acc.Issuer, 19)
			accountName := truncateString(acc.AccountName, 25)

			var countdownStyle lipgloss.Style
			if m.remaining <= 5 {
				countdownStyle = countdownCriticalStyle
			} else if m.remaining <= 10 {
				countdownStyle = countdownWarningStyle
			} else {
				countdownStyle = countdownNormalStyle
			}
			countdownText := fmt.Sprintf("%ds", m.remaining)

			row := lipgloss.JoinHorizontal(lipgloss.Left,
				issuerStyle.Render(issuer),
				accountNameStyle.Render(accountName),
				codeStyle.Render(m.codes[acc.Key()]),
				countdownStyle.Render(countdownText),
			)
			listBuilder.WriteString(row)
			if i < len(m.accounts)-1 {
				listBuilder.WriteString("\n")
			}
		}

		title := titleStyle.Render("🔒 2FA Authenticator")
		list := listContainerStyle.Render(listBuilder.String())
		help := helpStyle.Render("a: add account  •  q: quit")

		finalView = lipgloss.JoinVertical(lipgloss.Left, title, list, help)

	case stateAddingAccount:
		prompt := inputPromptStyle.Render("Paste your otpauth:// URI and press Enter:")
		input := m.textInput.View()
		help := helpStyle.Render("esc: cancel")

		content := lipgloss.JoinVertical(lipgloss.Left, prompt, "\n"+input, "\n"+help)
		if m.err != nil {
			errorMsg := errorStyle.Render("Error: " + m.err.Error())
			content = lipgloss.JoinVertical(lipgloss.Left, content, "\n\n"+errorMsg)
		}

		finalView = listContainerStyle.Render(content)
	}

	return docStyle.Render(finalView)
}

func doTick() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

func (m *model) generateCodes() tea.Cmd {
	return func() tea.Msg {
		newCodes := make(map[string]string)
		for _, acc := range m.accounts {
			code, err := totp.GenerateCode(acc.Secret, time.Now())
			if err != nil {
				return errorMsg{err: fmt.Errorf("failed to generate code for %s: %w", acc.Issuer, err)}
			}
			newCodes[acc.Key()] = code
		}
		return codesGeneratedMsg{codes: newCodes}
	}
}

func (m *model) addNewAccountCmd(uri string) tea.Cmd {
	return func() tea.Msg {
		otpURL, err := url.Parse(uri)
		if err != nil || otpURL.Scheme != "otpauth" {
			return errorMsg{fmt.Errorf("invalid otpauth:// URI format")}
		}
		secret := otpURL.Query().Get("secret")
		issuer := otpURL.Query().Get("issuer")
		if secret == "" {
			return errorMsg{fmt.Errorf("secret not found in URI")}
		}
		accountName := strings.TrimPrefix(otpURL.Path, "/totp/")
		accountName = strings.TrimPrefix(accountName, "/")
		if issuer == "" {
			parts := strings.SplitN(accountName, ":", 2)
			if len(parts) == 2 {
				issuer = parts[0]
			} else {
				issuer = "Unknown"
			}
		}
		newAcc := account.Account{Issuer: issuer, AccountName: accountName, Secret: secret}
		store := &storage.Store{FilePath: m.storagePath}
		data, err := store.Load()
		if err != nil {
			return errorMsg{fmt.Errorf("could not load store to save new account: %w", err)}
		}
		encryptedSecret, err := crypto.Encrypt([]byte(newAcc.Secret), m.encryptionKey)
		if err != nil {
			return errorMsg{err}
		}
		data.Accounts = append(data.Accounts, storage.StoredAccount{Issuer: newAcc.Issuer, AccountName: newAcc.AccountName, EncryptedSecret: encryptedSecret})
		if err := store.Save(data); err != nil {
			return errorMsg{err}
		}
		return accountAddedMsg{newAccount: newAcc}
	}
}

func truncateString(s string, maxLength int) string {
	if len(s) > maxLength {
		return s[:maxLength-3] + "..."
	}
	return s
}
