package account

type Account struct {
	Issuer      string
	AccountName string
	Secret      string
}

func (a Account) Key() string {
	return a.Issuer + ":" + a.AccountName
}
