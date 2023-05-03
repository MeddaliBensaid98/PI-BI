import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { User } from '../user';
import { catchError } from 'rxjs/operators';
import { of } from 'rxjs';
@Injectable({
  providedIn: 'root'
})


export class AuthService {
  private apiUrl = 'http://localhost:9912/api/user';

  constructor(private http: HttpClient) { }

  signup(user:any):Observable<any>
  {return this.http.post(`${this.apiUrl}/signup`,user, { observe: 'response' });}

  update(username: string,user:any):Observable<any>
  {return this.http.put(`${this.apiUrl}/update/${username}`,user);}

  login(user: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/login`, user, { observe: 'response' });
  }
  getUser(username: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/get/${username}`);
  }
}
