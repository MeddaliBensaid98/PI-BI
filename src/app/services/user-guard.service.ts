import { Injectable } from '@angular/core';
import{Router, ActivatedRouteSnapshot, CanActivate, RouterStateSnapshot,UrlTree} from '@angular/router';
import{Observable} from 'rxjs';
import { AuthService } from '../services/auth.service';

@Injectable({providedIn: 'root'})
export class UserGuardService   {
  constructor(private router:Router,private authService: AuthService)
  { }

  canActivate( next:ActivatedRouteSnapshot,state:RouterStateSnapshot):Observable<boolean |UrlTree> | Promise<boolean  |
  UrlTree> |boolean | UrlTree {

    if ((localStorage.getItem('user')==null) )
    {

      alert(' FORBIDDEN ACCESS  !!');
      this.router.navigate(['/']);

          return false;
     }





       return true;





}
}

