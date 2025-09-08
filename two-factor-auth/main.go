package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"syscall"

	"two-factor-auth/account"
	"two-factor-auth/crypto"
	"two-factor-auth/storage"
	"two-factor-auth/tui"

	"github.com/charmbracelet/lipgloss"
	"golang.org/x/term"
)

var (
	titleStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true).Margin(0, 0, 1, 0)
	errorStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Bold(true)
	inputPromptStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("11")).Bold(true)
)

func main() {
	store, err := storage.NewStore()
	if err != nil {
		log.Fatalf("Failed to initialize storage: %v", err)
	}

	var encryptionKey []byte
	var accounts []account.Account

	if !store.Exists() {
		encryptionKey, err = handleFirstRun(store)
		if err != nil {
			log.Fatalf("Setup failed: %v", err)
		}
	} else {
		encryptionKey, accounts, err = handleExistingUser(store)
		if err != nil {
			log.Fatalf(errorStyle.Render("Error: %v"), err)
		}
	}

	// Start the Bubble Tea UI
	if err := tui.Start(encryptionKey, accounts, store.Path()); err != nil {
		log.Fatalf("UI failed to start: %v", err)
	}
}

// handleFirstRun guides the user through creating a master password and initializing the secure store.
func handleFirstRun(store *storage.Store) ([]byte, error) {
	fmt.Println(titleStyle.Render("Go 2FA CLI Setup"))
	fmt.Println("No secrets file found. Let's create a secure one.")

	var password []byte
	var err error
	for {
		password, err = readPassword("Enter new master password: ")
		if err != nil {
			return nil, fmt.Errorf("failed to read password: %w", err)
		}
		confirmPassword, err := readPassword("Confirm master password: ")
		if err != nil {
			return nil, fmt.Errorf("failed to read password: %w", err)
		}
		if string(password) == string(confirmPassword) {
			break
		}
		fmt.Println(errorStyle.Render("Passwords do not match. Please try again."))
	}

	salt, err := crypto.GenerateSalt()
	if err != nil {
		return nil, fmt.Errorf("failed to generate salt: %w", err)
	}

	encryptionKey, err := crypto.GenerateKey(password, salt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate encryption key: %w", err)
	}

	initialData := &storage.StoredData{Salt: salt, Accounts: []storage.StoredAccount{}}
	if err := store.Save(initialData); err != nil {
		return nil, fmt.Errorf("failed to create secrets file: %w", err)
	}

	fmt.Println("Secrets file created successfully.")
	return encryptionKey, nil
}

// handleExistingUser prompts for the master password, loads, and decrypts the stored accounts.
func handleExistingUser(store *storage.Store) ([]byte, []account.Account, error) {
	data, err := store.Load()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load secrets file: %w", err)
	}

	password, err := readPassword(inputPromptStyle.Render("Enter master password: "))
	if err != nil {
		return nil, nil, fmt.Errorf("could not read password: %w", err)
	}

	encryptionKey, err := crypto.GenerateKey(password, data.Salt)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to derive key: %w", err)
	}

	// Decrypt accounts to verify the password before starting the UI.
	var accounts []account.Account
	for _, sa := range data.Accounts {
		secret, err := crypto.Decrypt(sa.EncryptedSecret, encryptionKey)
		if err != nil {
			return nil, nil, fmt.Errorf("decryption failed. Please check your password")
		}
		accounts = append(accounts, account.Account{
			Issuer:      sa.Issuer,
			AccountName: sa.AccountName,
			Secret:      string(secret),
		})
	}
	return encryptionKey, accounts, nil
}

// readPassword reads a password from stdin without echoing it.
func readPassword(prompt string) ([]byte, error) {
	fmt.Print(prompt)
	password, err := term.ReadPassword(int(syscall.Stdin))
	fmt.Println() // Add a newline after the user presses enter.
	// Clear stdin buffer in case of extra characters
	bufio.NewReader(os.Stdin).Reset(os.Stdin)
	return password, err
}
